#include <cmath>
#include "determ.hpp"
#include "num_sim.hpp"
#include "model_impl.hpp"
#include <limits>
#include <iostream>
#include <cassert>

CUDA_AGNOSTIC
Minutes dense::Deterministic_Simulation::age_by (Minutes duration) {
  assert(duration > 0 && t > 0 && _step_size > 0);
  dense::Natural steps = (duration /*+ std::remainder(t, _step_size)*/) / Minutes{ _step_size };
  auto start = std::chrono::high_resolution_clock::now();
  Simulation::step(true);
  for (dense::Natural s = 0; s < steps; ++s) {
    step();
    Simulation::step(false);
  }
  auto finish = std::chrono::high_resolution_clock::now();
  std::cout<< "steps per second: "<<Simulation::get_performance(finish - start)<<std::endl;
  return Simulation::age_by(duration);
}

CUDA_AGNOSTIC
void dense::Deterministic_Simulation::update_concentrations(dense::Natural cell, SpecieRates const& rates) {
    for (int i=0; i< NUM_SPECIES; i++){
        auto curr_rate = rates[i];
        _baby_cl.row_at(i, 1)[cell] = _baby_cl.row_at(i, 0)[cell] + _step_size * curr_rate;
    }
}

dense::Deterministic_Simulation::Deterministic_Simulation(const Parameter_Set& ps, Real* pnFactorsPert, Real** pnFactorsGrad,
                    Minutes step_size, std::vector<Real> conc, NGraph::Graph adj_graph, dense::Natural num_grow_cell) :
    Simulation(ps, std::move(adj_graph), pnFactorsPert, pnFactorsGrad, num_grow_cell), 
    Numerical_Integration(NUM_DELAY_REACTIONS, cell_count(), step_size, *this) {
      //Copy and normalize _delays into _intDelays
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

CUDA_AGNOSTIC
dense::Deterministic_Simulation::SpecieRates dense::Deterministic_Simulation::calculate_concentrations(dense::Natural cell) {
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
void dense::Deterministic_Simulation::step() {
    int c = 0;
    for (dense::Natural k = 0; k < dense::Natural(physical_cells_id().size()); k++) {
      if(physical_cells_id()[k] >= 0){ //check if the cell has been removed from the simulation
        update_concentrations(c, calculate_concentrations(c));
      }
      c++;
    }
    _j++;
    _baby_cl.advance();
}
