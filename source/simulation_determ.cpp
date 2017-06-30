#include <cmath>
#include "simulation_determ.hpp"
#include "cell_param.hpp"
#include "model_impl.hpp"
#include <limits>
#include <iostream>

typedef std::numeric_limits<double> dbl;
using namespace std;

void simulation_determ::simulate(){
	RATETYPE analysis_chunks = time_total/analysis_gran;
 	RATETYPE total_step = analysis_gran/_step_size;
	for (int c = 0; c<analysis_chunks;c++){
        Context context(*this,0);
        notify(context);
        
        for (int i = 0; i<total_step; i++) {
			execute();
		}
	}
	Context context(*this,0);
	notify(context,true);
}

void simulation_determ::execute(){
    //concentration cl;
    //Rates rates;
    //int steps_elapsed = steps_split; // Used to determine when to split a column of cells
    //update_rates(rs, active_start); // Update the active rates based on the base rates, perturbations, and gradients
    
    //int j;
    //bool past_induction = false; // Whether we've passed the point of induction of knockouts or overexpression
    //bool past_recovery = false; // Whether we've recovered from the knockouts or overexpression
        
    
    //where to keep the birth and parent information
    //copy_records(_contexts, _baby_j, _time_prev); // Copy each cell's birth and parent so the records are accessible at every time step
    //cout.precision(dbl::max_digits10);
    //cout<< _j<< " "<<_baby_cl[ph1][_j][0]<<endl;
    // Iterate through each extant cell or context
    for (int k = 0; k < _cells_total; k++) {
        //if (_width_current == _width_total || k % _width_total <= 10) { // Compute only existing (i.e. already grown)cells
                // Calculate the cell indices at the start of each mRNA and protein's dela
            Context c(*this, k);
            int old_cells_mrna[NUM_SPECIES];
            int old_cells_protein[NUM_SPECIES]; // birth and parents info are kept elsewhere now
            //calculate_delay_indices(_baby_cl, _baby_j, _j, k, _rates, old_cells_mrna, old_cells_protein);

            // Perform biological calculations
            c.updateCon(c.calculateRatesOfChange());
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

void simulation_determ::initialize(){
    simulation_base::initialize();
    _baby_cl.initialize();
    //Copy and normalize _delays into _intDelays
    for (int i = 0; i < NUM_DELAY_REACTIONS; i++) {
      for (int j = 0; j < _cells_total; j++) {
        _intDelays[i][j] = _delays[i][j] / _step_size;
      }
    }
}
