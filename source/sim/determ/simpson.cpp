#include <cmath>
#include "simpson.hpp"
#include "num_sim.hpp"
#include "model_impl.hpp"
#include "baby_cl.hpp"
#include <limits>
#include <iostream>
#include <cassert>

dense::Simpson_Simulation::Simpson_Simulation(const Parameter_Set& ps, Real* pnFactorsPert, Real** pnFactorsGrad,
                    Minutes step_size, std::vector<Real> conc, NGraph::Graph adj_graph) :
    Simulation(ps, std::move(adj_graph), pnFactorsPert, pnFactorsGrad),
    Numerical_Integration(NUM_DELAY_REACTIONS, cell_count(), step_size, *this, 1) {
      //Copy and normalize _delays into _intDelays
      std::vector<std::vector<double>>* v_1 = new std::vector<std::vector<double>> (cell_count(), std::vector<double> (NUM_SPECIES, 0));
      std::vector<std::vector<double>>* v_2 = new std::vector<std::vector<double>> (cell_count(), std::vector<double> (NUM_SPECIES, 0));
      _prev_rates = *v_1;
      _last_rates = *v_2;
      delete v_1;
      delete v_2;
      for (int i = 0; i < NUM_DELAY_REACTIONS; i++) {
        for (dense::Natural j = 0; j < cell_count(); ++j) {
          _intDelays[i][j] = cell_parameters_[NUM_REACTIONS+i][j] / _step_size;
        }
      }
      if (!conc.empty()) {
        for (int specie = 0; specie < NUM_SPECIES; specie++) {
          int specie_size = _baby_cl.get_species_size(specie);
          for (int time = 0; time < specie_size; time++){
            for(dense::Natural cell = 0; cell < cell_count(); cell++){
              _baby_cl.row_at(specie,time)[cell] = conc[specie];
            }
          }
        }
      }
    }




void dense::Simpson_Simulation::step() {
    for (dense::Natural k = 0; k < cell_count(); k++) {
      update_concentrations(k, calculate_concentrations(k));
    }
    if (!_first_point_calculated) {
      _first_point_calculated = true;
    } else if (!_second_point_calculated) {
      _second_point_calculated = true;
      _curr_coeff = 4;
    } else if (_curr_coeff == 4) {
      _curr_coeff = 2;
    } else {
      _curr_coeff = 4;
    }
    _j++;
    _baby_cl.advance();
}

CUDA_AGNOSTIC
void dense::Simpson_Simulation::update_concentrations(dense::Natural cell, SpecieRates const& rates) {
  //TODO: Implement Simpson's rule in this function
  //      it is currently left as the same implementation as Deterministic_Simulation,
  //      just to see what method was previously used.

    //Runs through and performs one step of Simpson's Rule to each specie
    for (int i = 0; i < NUM_SPECIES; i++) {
      //_curr_coeff is '1' for the first two steps, we just use Euler's Method
      double curr_rate = rates[i];
      if (!_second_point_calculated) {
        _baby_cl.row_at(i, 1)[cell] = _baby_cl.row_at(i, 0)[cell] + _step_size * curr_rate;
        if (_first_point_calculated) {
          _last_rates[cell][i] = _prev_rates[cell][i];
        }
        _prev_rates[cell][i] = curr_rate;
      } else {
        _baby_cl.row_at(i, 1)[cell] = _baby_cl.row_at(i, -1)[cell] + (_step_size/3)*(6.5*curr_rate - _prev_rates[cell][i] + 0.5*_last_rates[cell][i]);
        _last_rates[cell][i] = _prev_rates[cell][i];
        _prev_rates[cell][i] = curr_rate;
      }
    }
    //Updating what has been done.
    //These if-else statements are ordered in priority

}


CUDA_AGNOSTIC
dense::Simpson_Simulation::SpecieRates dense::Simpson_Simulation::calculate_concentrations(dense::Natural cell) {
    //Step 1: for each reaction, compute reaction rate
    CUDA_Array<Real, NUM_REACTIONS> reaction_rates;
    #define REACTION(name) reaction_rates[name] = dense::model::reaction_##name.active_rate(Context(*this, cell));
        #include "reactions_list.hpp"
    #undef REACTION

    //Step 2: allocate specie concentration rate change array
    SpecieRates specie_deltas{};

    //Step 3: for each reaction rate, for each specie it affects, accumulate its contributions
    #define REACTION(name) \
    const reaction<name>& r##name = dense::model::reaction_##name; \
    for (int j = 0; j < r##name.getNumDeltas(); j++) { \
        specie_deltas[delta_ids_##name[j]] += reaction_rates[name]*deltas_##name[j]; \
    }
    #include "reactions_list.hpp"
    #undef REACTION

    return specie_deltas;
}


CUDA_AGNOSTIC
Minutes dense::Simpson_Simulation::age_by (Minutes duration) {
  //TODO: This should be able to stay as is, but confirm that the only real difference is in the definition of Step()
  assert(duration > (Minutes)0 && _step_size > 0);
  dense::Natural steps = (duration /*+ std::remainder(t, _step_size)*/) / Minutes{ _step_size };
  for (dense::Natural s = 0; s < steps; ++s) {
    step();
  }
  return Simulation::age_by(duration);
}
