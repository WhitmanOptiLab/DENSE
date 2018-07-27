#include <cmath>
#include "determ.hpp"
#include "model_impl.hpp"
#include <limits>
#include <iostream>
#include <cassert>

void dense::Deterministic_Simulation::simulate_for (Real duration) {
  assert(duration > 0 && t > 0 && _step_size > 0);
  dense::Natural steps = (duration /*+ std::remainder(t, _step_size)*/) / _step_size;
  for (dense::Natural s = 0; s < steps; ++s) {
    step();
  }
  age_ += duration;
}

void dense::Deterministic_Simulation::update_concentrations(dense::Natural cell, SpecieRates const& rates) {
    for (int i=0; i< NUM_SPECIES; i++){
        auto curr_rate = rates[i];
        _baby_cl[i][1][cell] = _baby_cl[i][0][cell] + _step_size * curr_rate;
    }
}

dense::Deterministic_Simulation::Deterministic_Simulation(const Parameter_Set& ps, Real* pnFactorsPert, Real** pnFactorsGrad, int cells_total, int width_total,
                    Real step_size) :
    Simulation(ps, cells_total, width_total, pnFactorsPert, pnFactorsGrad), _intDelays(width_total, cells_total),
    _baby_cl(*this), _step_size(step_size), _j(0), _num_history_steps(2) {
      _baby_cl.initialize();
      //Copy and normalize _delays into _intDelays
      for (int i = 0; i < NUM_DELAY_REACTIONS; i++) {
        for (dense::Natural j = 0; j < _cells_total; ++j) {
          _intDelays[i][j] = _cellParams[NUM_REACTIONS+i][j] / _step_size;
        }
      }
    }

CUDA_HOST CUDA_DEVICE
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

void dense::Deterministic_Simulation::step() {
    //concentration cl;
    //Rates rates;
    //int steps_elapsed = steps_split; // Used to determine when to split a column of cells
    //update_rates(rs, active_start); // Update the active rates based on the base rates, perturbations, and gradients

    //int j;
    //bool past_induction = false; // Whether we've passed the point of induction of knockouts or overexpression
    //bool past_recovery = false; // Whether we've recovered from the knockouts or overexpression


    //where to keep the birth and parent information
    //copy_records(_contexts, _baby_j, _time_prev); // Copy each cell's birth and parent so the records are accessible at every time step
    //cout.precision(std::numeric_limits<double>::max_digits10);
    //cout<< _j<< " "<<_baby_cl[ph1][_j][0]<<endl;
    // Iterate through each extant cell or context
    for (dense::Natural k = 0; k < _cells_total; k++) {
        //if (_width_current == _width_total || k % _width_total <= 10) { // Compute only existing (i.e. already grown)cells
                // Calculate the cell indices at the start of each mRNA and protein's dela
            //int old_cells_mrna[NUM_SPECIES];
            //int old_cells_protein[NUM_SPECIES]; // birth and parents info are kept elsewhere now
            //calculate_delay_indices(_baby_cl, _baby_j, _j, k, _rates, old_cells_mrna, old_cells_protein);

            // Perform biological calculations
            update_concentrations(k, calculate_concentrations(k));
        //}
    }

    // Check to make sure the numbers are still valid
    /*
    if (any_less_than_0(_baby_cl, _baby_j) || concentrations_too_high(_baby_cl, _baby_j, max_con_thresh)) {
        //return false;
        //printf "Concentration too high or below zero. Exiting."
        exit(0);
    }
     */

    // Update the active record data and split counter
    //steps_elapsed++;
    //baby_cl.active_start_record[baby_j] = active_start;
    //baby_cl.active_end_record[baby_j] = active_end;

    _j++;
    //Advance the current timestep
    _baby_cl.advance();
    //print the concentration level of mh1 for cell 1

}
