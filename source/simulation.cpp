
#include "simulation.hpp"
#include "rates.hpp"
#include "contexts.hpp"
#include "model_impl.hpp"

#include <iostream>

//declare reaction inits here
#define REACTION(name) \
  template<> \
  reaction< name >::reaction() : \
    num_inputs(num_inputs_##name), num_outputs(num_outputs_##name), \
    in_counts(in_counts_##name), inputs(inputs_##name), \
    out_counts(out_counts_##name), outputs(outputs_##name) {}
LIST_OF_REACTIONS
#undef REACTION

//A quick test case to make sure all reaction rates are defined by link time
void simulation::test_sim() {
  Context<double> c;

  double sum_rates = 0.0;
#define REACTION(name) sum_rates += _model.reaction_##name.active_rate(c);
  LIST_OF_REACTIONS
#undef REACTION
  std::cout << "If you're seeing this, simulation.cpp compiles correctly:" 
            << sum_rates << std::endl;
}


void simulation::model(){
    //concentration cl;
    Rates rates;
    int steps_elapsed = steps_split; // Used to determine when to split a column of cells
    //update_rates(rs, active_start); // Update the active rates based on the base rates, perturbations, and gradients
    
    //Context<double> contexts[]= {};
    //int j;
    //int baby_j;
    //bool past_induction = false; // Whether we've passed the point of induction of knockouts or overexpression
    //bool past_recovery = false; // Whether we've recovered from the knockouts or overexpression
    //for (j = time_start; j < time_end; j++) {
        
        
    for (int i=0; i< sizeof(baby_j);i++){
        time_prev[i]= WRAP(baby_j[i]-1, rates.delay_size[i]);
    }
        //int time_prev = WRAP(baby_j - 1, sd.max_delay_size); // Time is cyclical, so time_prev may not be baby_j - 1
    copy_records(baby_cl, baby_j, time_prev); // Copy each cell's birth and parent so the records are accessible at every time step
        
        
    // Iterate through each extant cell
    for (int k = 0; k < cells_total; k++) {
        if (width_current == width_total || k % width_total <= active_start) { // Compute only existing (i.e. already grown)cells
                // Calculate the cell indices at the start of each mRNA and protein's dela
            int old_cells_mrna[NUM_INDICES];
            int old_cells_protein[NUM_INDICES];
            calculate_delay_indices(sd, baby_cl, baby_j, j, k, rs.rates_active, old_cells_mrna, old_cells_protein);
                
            // Perform biological calculations
            #define REACTION(name);
                LIST_OF_REACTIONS
            #undef REACTION
        }
    }
    // Check to make sure the numbers are still valid
    if (any_less_than_0(baby_cl, baby_j) || concentrations_too_high(baby_cl, baby_j, max_con_thresh)) {
        return false;
    }
    
    // Update the active record data and split counter
    steps_elapsed++;
    //baby_cl.active_start_record[baby_j] = sd.active_start;
    //baby_cl.active_end_record[baby_j] = sd.active_end;
    
    // Copy from the simulating cl to the analysis cl
    if (j % big_gran == 0) {
        baby_to_cl(baby_cl, cl, baby_j, j / big_gran);
    }
        
    //}
    
    // Copy the last time step from the simulating cl to the analysis cl and mark where the simulating cl left off time-wise
    baby_to_cl(cl,);
    //sd.time_baby = baby_j;
    //return true;

}



void simulation::baby_to_cl(concentration cl){
    for (int i = 0; i < cl.num_con_levels; i++) {
        for (int k = 0; k < cl.cells; k++) {
            cl.cons[i][time][k] = baby_cl.cons[i][baby_time][k];
        }
    }
    cl.active_start_record[time] = baby_cl.active_start_record[baby_time];
    cl.active_end_record[time] = baby_cl.active_end_record[baby_time];


}


inline void simulation::copy_records (con_levels& cl, vector<int> time, vector<int> time_prev) {
    for (int k = 0; k < sd.cells_total; k++) {
        cl.cons[BIRTH][time][k] = cl.cons[BIRTH][time_prev][k];
        cl.cons[PARENT][time][k] = cl.cons[PARENT][time_prev][k];
    }
}


inline bool simulation::any_less_than_0 (con_levels& cl, vector<int> time) {
    for (int i = MIN_CON_LEVEL; i <= MAX_CON_LEVEL; i++) {
        if (cl.cons[i][time][0] < 0) { // This checks only the first cell
            return true;
        }
    }
    return false;
}

inline bool simulation::concentrations_too_high (con_levels& cl, vector<int> time, double max_con_thresh) {
    if (max_con_thresh != INFINITY) {
        for (int i = MIN_CON_LEVEL; i <= MAX_CON_LEVEL; i++) {
            if (cl.cons[i][time][0] > max_con_thresh) { // This checks only the first cell
                return true;
            }
        }
    }
    return false;
}
