#include <cmath>
#include "determ.hpp"
#include "model_impl.hpp"
#include <limits>
#include <iostream>
#include <cassert>

CUDA_AGNOSTIC
Minutes dense::Deterministic_Simulation::age_by (Minutes duration) {
  assert(duration > 0 && t > 0 && _step_size > 0);
  dense::Natural steps = (duration /*+ std::remainder(t, _step_size)*/) / Minutes{ _step_size };
  for (dense::Natural s = 0; s < steps; ++s) {
    step();
  }
  return Simulation::age_by(duration);
}

CUDA_AGNOSTIC
void dense::Deterministic_Simulation::update_concentrations(dense::Natural cell, SpecieRates const& rates) {
    for (int i=0; i< NUM_SPECIES; i++){
        auto curr_rate = rates[i];
        _baby_cl.row_at(i, 1)[cell] = _baby_cl.row_at(i, 0)[cell] + _step_size * curr_rate;
    }
}

dense::Deterministic_Simulation::Deterministic_Simulation(const Parameter_Set& ps, Real* pnFactorsPert, Real** pnFactorsGrad, int width_total,
                    Minutes step_size, std::vector<Real> conc, NGraph::Graph* adj_graph) :
    Simulation(ps, adj_graph, pnFactorsPert, pnFactorsGrad), 
    _intDelays(NUM_DELAY_REACTIONS, cell_count()),
    _step_size{step_size / Minutes{1}}, _j(0), _num_history_steps(2), _baby_cl(*this) {
      //Copy and normalize _delays into _intDelays
      for (int i = 0; i < NUM_DELAY_REACTIONS; i++) {
        for (dense::Natural j = 0; j < cell_count(); ++j) {
          _intDelays[i][j] = cell_parameters_[NUM_REACTIONS+i][j] / _step_size;
        }
      }
      //Initialize the baby_cl concentrations with 0s or user specified amount
      for (int specie = 0; specie < NUM_SPECIES; specie++) {
        int specie_size = _baby_cl.get_species_size(specie);
        for (int time = 0; time < specie_size; time++){
          for(dense::Natural cell = 0; cell < adj_graph->num_vertices(); cell++){
            _baby_cl.row_at(specie,time)[cell] = conc[specie];
          }
        }
      }
    }

CUDA_AGNOSTIC
dense::Deterministic_Simulation::SpecieRates dense::Deterministic_Simulation::calculate_concentrations(dense::Natural cell) {
    //Step 1: for each reaction, compute reaction rate
    CUDA_Array<Real, NUM_REACTIONS> reaction_rates;
    #define REACTION(name) reaction_rates[name] = dense::model::reaction_##name.active_rate(Context(*this));
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
void dense::Deterministic_Simulation::step() {
    for (dense::Natural k = 0; k < cell_count(); k++) {
      update_concentrations(k, calculate_concentrations(k));
    }
    _j++;
    _baby_cl.advance();
}
