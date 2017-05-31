#ifndef SIMULATION_HPP
#define SIMULATION_HPP
#include "observable.hpp"
#include "param_set.hpp"
#include "model.hpp"
#include "specie.hpp"
#include "cell_param.hpp"
#include "reaction.hpp"
#include "concentration_level.hpp"
#include "baby_cl.hpp"
#include <vector>
#include <array>
using namespace std;

/* simulation contains simulation data, partially taken from input_params and partially derived from other information
	notes:
 There should be only one instance of simulation at any time.
	todo:
 */


typedef cell_param<NUM_REACTIONS> Rates;
typedef cell_param<NUM_DELAY_REACTIONS> Delays;
typedef cell_param<NUM_CRITICAL_SPECIES> CritValues;

class simulation : public Observable{
  
 public:
    class Context {
        //FIXME - want to make this private at some point
    public:
        typedef CPUGPU_TempArray<RATETYPE, NUM_SPECIES> SpecieRates;
        const int _cell;
        simulation& _simulation;
        double _avg;
        CPUGPU_FUNC
        Context(simulation& sim, int cell) : _simulation(sim),_cell(cell) { }
        CPUGPU_FUNC
        RATETYPE calculateNeighborAvg(specie_id sp, int delay = 0) const;
        CPUGPU_FUNC
        void updateCon(const SpecieRates& rates);
        CPUGPU_FUNC
        const SpecieRates calculateRatesOfChange();
        CPUGPU_FUNC
        RATETYPE getCon(specie_id sp, int delay = 1) const {
            int modified_step = _simulation._baby_j[sp] + 1 - delay;
            return _simulation._baby_cl[sp][modified_step][_cell];
        }
        CPUGPU_FUNC
        RATETYPE getCritVal(critspecie_id rcritsp) const {
            return _simulation._critValues[rcritsp][_cell];
        }
        CPUGPU_FUNC
        RATETYPE getRate(reaction_id reaction) const {
            return _simulation._rates[reaction][_cell];
        }
        CPUGPU_FUNC
        RATETYPE getDelay(delay_reaction_id delay_reaction) const{
            return _simulation._delays[delay_reaction][_cell]/_simulation._step_size;
        }
    };
  // PSM stands for Presomitic Mesoderm (growth region of embryo)

  // Sizes
  int _width_total; // The width in cells of the PSM
  int _width_initial; // The width in cells of the PSM before anterior growth
  int _width_current; // The width in cells of the PSM at the current time step
  int height; // The height in cells of the PSM
  int cells; // The number of cells in the simulation
  int _cells_total; // The total number of cells of the PSM (total width * total height)

  // Times and timing
  RATETYPE _step_size; // The step size in minutes
  RATETYPE time_total;
  RATETYPE analysis_gran;
  int steps_total; // The number of time steps to simulate (total time / step size)
  int steps_split; // The number of time steps it takes for cells to split
  int steps_til_growth; // The number of time steps to wait before allowing cells to grow into the anterior PSM
  bool no_growth; // Whether or not the simulation should rerun with growth

  // Granularities
  int _big_gran; // The granularity in time steps with which to analyze and store data
  int small_gran; // The granularit in time steps with which to simulate data

  // Cutoff values
  double max_con_thresh; // The maximum threshold concentrations can reach before the simulation is prematurely ended
  int max_delay_size; // The maximum number of time steps any delay in the current parameter set takes plus 1 (so that baby_cl and each mutant know how many minutes to store)
  
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
  Rates _rates;
  Delays _delays;
  CritValues _critValues;
  Concentration_level _cl;
  baby_cl _baby_cl;
  //Context<double> _contexts;
  CPUGPU_TempArray<int,NUM_SPECIES> _baby_j;
  //int* _delay_size;
  //int* _time_prev;
  int _j;
  CPUGPU_TempArray<int, 6>* _neighbors;
  //double* _sets;
  int _NEIGHBORS_2D;
  int _num_history_steps; // how many steps in history are needed for this numerical method
  //int* _relatedReactions[NUM_SPECIES];
  int max_delays[NUM_SPECIES];  // The maximum number of time steps that each specie might be accessed in the past
    

    
  simulation(const model& m, const param_set& ps, int cells_total, int width_total, RATETYPE step_size, RATETYPE analysis_interval, RATETYPE sim_time) : _cells_total(cells_total),_width_total(width_total),_width_initial(width_total),_width_current(width_total), _parameter_set(ps), _model(m), _rates(*this, cells_total), _delays(*this, cells_total), _critValues(*this, cells_total),_cl(*this), _baby_cl(*this), _neighbors(new CPUGPU_TempArray<int, 6>[_cells_total]), _step_size(step_size){
    //,_baby_j(NUM_REACTIONS), _time_prev(NUM_REACTIONS), _contexts(cells), _rates()
      _j =0 ;
      analysis_gran = analysis_interval;
      time_total = sim_time;
      for (int i = 0; i < NUM_SPECIES; i++) {
        _baby_j[i] = 0;
      }
      _NEIGHBORS_2D = 6;
      _big_gran = 1;
      _num_history_steps = 2;
      cout << "no seg fault2"<<endl;
  }
  ~simulation() {delete[] _neighbors; }
  void test_sim();
  void execute();
    void baby_to_cl(baby_cl& baby_cl, Concentration_level& cl, int time, int* baby_times){
        int baby_time = 0;
        //cout<<"09"<<endl;
        for (int i = 0; i <= NUM_SPECIES; i++) {
            //cout<<"10"<<endl;
            baby_time = baby_times[i];
            for (int k = 0; k < _cells_total; k++) {
                //cout<<"11"<<endl;
                RATETYPE temp =baby_cl[i][baby_time][k];
                //cout<<"12"<<endl;
                cl[i][time][k] = temp;
            }
        }
    }
  void copy_records(Concentration_level& cl, int* time, int* time_prev);
  bool any_less_than_0(baby_cl& baby_cl, int* times);
  bool concentrations_too_high (baby_cl& baby_cl, int* time, double max_con_thresh);
#ifdef __CUDACC__
  __host__ __device__
#endif
    void calculate_delay_indices(baby_cl& baby_cl, int* baby_time, int time, int cell_index, Rates& rs, int old_cells_mrna[], int old_cells_protein[]){
        for (int l = 0; l < NUM_SPECIES; l++) {
            old_cells_mrna[l] = cell_index;
            old_cells_protein[l] = cell_index;
        }
    }
    
  void initialize();
#ifdef __CUDACC__
    __host__ __device__
#endif
    void calc_neighbor_2d(){
        for (int i = 0; i < _cells_total; i++) {
            if (i % 2 == 0) {																		// All even column cells
                _neighbors[i][0] = (i - _width_total + _cells_total) % _cells_total;			// Top
                _neighbors[i][1] = (i - _width_total + 1 + _cells_total) % _cells_total;		// Top-right
                _neighbors[i][2] = (i + 1) % _cells_total;											// Bottom-right
                _neighbors[i][3] = (i + _width_total) % _cells_total;								// Bottom
                if (i % _width_total == 0) {														// Left edge
                    _neighbors[i][4] = i + _width_total - 1;										// Bottom-left
                    _neighbors[i][5] = (i - 1 + _cells_total) % _cells_total;						// Top-left
                } else {																			// Not a left edge
                    _neighbors[i][4] = i - 1;															// Bottom-left
                    _neighbors[i][5] = (i - _width_total - 1 + _cells_total) % _cells_total;	// Top-left
                }
            } else {																				// All odd column cells
                _neighbors[i][0] = (i - _width_total + _cells_total) % _cells_total;			// Top
                if (i % _width_total == _width_total - 1) {											// Right edge
                    _neighbors[i][1] = i - _width_total + 1;										// Top-right
                    _neighbors[i][2] = (i + 1) % _cells_total;										// Bottom-right
                } else {																			// Not a right edge
                    _neighbors[i][1] = i + 1;															// Top-right
                    _neighbors[i][2] = (i + _width_total + 1 + _cells_total) % _cells_total;	// Nottom-right
                }																					// All odd column cells
                _neighbors[i][3] = (i + _width_total) % _cells_total;								// Bottom
                _neighbors[i][4] = (i + _width_total - 1) % _cells_total;							// Bottom-left
                _neighbors[i][5] = (i - 1 + _cells_total) % _cells_total;							// Top-left
            }
        }
    }
    
  void set_test_data();
    void simulate();
    void print_delay(){
        cout<<"delay for mh1 ";
        for (int i =0; i<_cells_total;i++){
            cout<< _delays[dreact_mh1_synthesis][i]<< " ";
        }
        cout<<endl;
    }
 protected:
  void calc_max_delays();
};
#endif

