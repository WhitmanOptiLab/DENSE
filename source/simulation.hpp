#ifndef SIMULATION_HPP
#define SIMULATION_HPP

#include "param_set.hpp"
#include "model.hpp"


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
  int cells; // The number of cells in the simulation
  int cells_total; // The total number of cells of the PSM (total width * total height)

  // Neighbors and boundaries
  //array2D<int> neighbors; // An array of neighbor indices for each cell position used in 2D simulations (2-cell and 1D calculate these on the fly)
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

  const param_set& _parameter_set;
  const model& _model;
    Rates& _rates;
    Concentration_level& _cl;
    Concentration_level& _babyl_cl;
    vector<Context>& _contexts;
    vector<int>& _baby_j;
    vector<int>& _delay_size;
    vector<int>& _time_prev;
    int _j;
    int* _neighbors;
    double** _sets;
    int NEIGHBORS_2D;
    
 public:
  simulation(const model& m, const param_set& ps) : _parameter_set(ps), _model(m)
    //,_baby_j(NUM_REACTIONS), _time_prev(NUM_REACTIONS), _contexts(cells), _rates()
    { }
  void test_sim();
    void model();
    void baby_to_cl(Concentration_level& _cl);
    void copy_records(Concentration_level& _cl, vector<int>& time, vector<int> time_prev);
    bool any_less_than_0(Concentration_level& baby_cl, vector<int>& time);
    bool concentrations_too_high (con_levels& baby_cl, vector<int> time, double max_con_thresh);
    void calculate_delay_indices(Concentration_level& baby_cl, vector<int> baby_time, vector<int> time, int cell_index, Rates& rs, int old_cells_mrna[], int old_cells_protein[]));
    void initialize();
    void calc_neighbor_2d();
};
#endif

