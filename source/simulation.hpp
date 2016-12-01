#ifndef SIMULATION_HPP
#define SIMULATION_HPP



using namespace std;

/* simulation contains simulation data, partially taken from input_params and partially derived from other information
	notes:
 There should be only one instance of simulation at any time.
	todo:
 */

class simulation{
    
private:
    // Times and timing
    double step_size; // The step size in minutes
    int time_total; // The number of minutes to run for
    int steps_total; // The number of time steps to simulate (total time / step size)
    int steps_split; // The number of time steps it takes for cells to split
    int steps_til_growth; // The number of time steps to wait before allowing cells to grow into the anterior PSM
    bool no_growth; // Whether or not the simulation should rerun with growth
    
    // Granularities
    int big_gran; // The granularity in time steps with which to analyze and store data
    int small_gran; // The granularit in time steps with which to simulate data
    
    // Cutoff values
    double max_con_thresh; // The maximum threshold concentrations can reach before the simulation is prematurely ended
    int max_delay_size; // The maximum number of time steps any delay in the current parameter set takes plus 1 (so that baby_cl and each mutant know how many minutes to store)
    
    // Sizes
    int width_total; // The width in cells of the PSM
    int width_initial; // The width in cells of the PSM before anterior growth
    int width_current; // The width in cells of the PSM at the current time step
    int height; // The height in cells of the PSM
    int cells_total; // The total number of cells of the PSM (total width * total height)
    
    // Neighbors and boundaries
    array2D<int> neighbors; // An array of neighbor indices for each cell position used in 2D simulations (2-cell and 1D calculate these on the fly)
    int active_start; // The start of the active portion of the PSM
    int active_end; // The end of the active portion of the PSM
    
    // PSM section and section-specific times
    int section; // Posterior or anterior (sec_post or sec_ant)
    int time_start; // The start time (in time steps) of the current simulation
    int time_end; // The end time (in time steps) of the current simulation
    int time_baby; // Time 0 for baby_cl at the end of a simulation
    
    // Mutants and condition scores
    int num_active_mutants; // The number of mutants to simulate for each parameter set
    double max_scores[NUM_SECTIONS]; // The maximum score possible for all mutants for each testing section
    double max_score_all; // The maximum score possible for all mutants for all testing sections

public:
    
};
#endif

