#ifndef SIM_BASE_HPP
#define SIM_BASE_HPP

#include "util/common_utils.hpp"
#include "core/context.hpp"
#include "core/observable.hpp"
#include "core/param_set.hpp"
#include "core/model.hpp"
#include "core/specie.hpp"
#include "cell_param.hpp"
#include "core/reaction.hpp"
#include <vector>


/* simulation contains simulation data, partially taken from input_params and partially derived from other information
 */
/* SIMULATION_BASE
 * superclass for simulation_determ and simulation_stoch
 * inherits from Observable, can be observed by Observer object
*/
class simulation_base : public Observable{
  
  public:
  // Sizes
  int _width_total; // The maximum width in cells of the PSM
  int circumf; // The current width in cells
  //int _width_initial; // The width in cells of the PSM before anterior growth
  //int _width_current; // The width in cells of the PSM at the current time step
  //int height; // The height in cells of the PSM
  //int cells; // The number of cells in the simulation
  int _cells_total; // The total number of cells of the PSM (total width * total height)

  // Times and timing
  RATETYPE time_total;
  RATETYPE analysis_gran;
  //int steps_total; // The number of time steps to simulate (total time / step size)
  //int steps_split; // The number of time steps it takes for cells to split
  //int steps_til_growth; // The number of time steps to wait before allowing cells to grow into the anterior PSM
  //bool no_growth; // Whether or not the simulation should rerun with growth

  // Neighbors and boundaries
  //array2D<int> neighbors; // An array of neighbor indices for each cell position used in 2D simulations (2-cell and 1D calculate these on the fly)
  //int active_start; // The start of the active portion of the PSM
  //int active_end; // The end of the active portion of the PSM
  CPUGPU_TempArray<int, 6>* _neighbors;
  
  // PSM section and section-specific times
  //int section; // Posterior or anterior (sec_post or sec_ant)
  //int time_start; // The start time (in time steps) of the current simulation
  //int time_end; // The end time (in time steps) of the current simulation

  // Mutants and condition scores
  //int num_active_mutants; // The number of mutants to simulate for each parameter set
  //double max_scores[NUM_SECTIONS]; // The maximum score possible for all mutants for each testing section
  //double max_score_all; // The maximum score possible for all mutants for all testing sections

  param_set const& _parameter_set;
  model const& _model;
  RATETYPE* factors_perturb;
  RATETYPE** factors_gradient;
  cell_param<NUM_REACTIONS+NUM_DELAY_REACTIONS+NUM_CRITICAL_SPECIES> _cellParams;
  int *_numNeighbors;
  //CPUGPU_TempArray<int,NUM_SPECIES> _baby_j;
  //int* _delay_size;
  //int* _time_prev;
  //double* _sets;
  //int _NEIGHBORS_2D;
  RATETYPE max_delays[NUM_SPECIES];  // The maximum number of time steps that each specie might be accessed in the past
    

  /*
   * CONSTRUCTOR
   * arg "m": assiged to "_model", used to access user-inputted active rate functions
   * arg "ps": assiged to "_parameter_set", used to access user-inputted rate constants, delay times, and crit values
   * arg "cells_total": the maximum amount of cells to simulate for (initial count for non-growing tissues)
   * arg "width_total": the circumference of the tube, in cells
   * arg "analysis_interval": the interval between notifying observers for data storage and analysis, in minutes
   * arg "sim_time": the total time to simulate for, in minutes
  */
  simulation_base(model const& m, param_set const& ps, RATETYPE* pnFactorsPert, RATETYPE** pnFactorsGrad, int cells_total, int width_total, RATETYPE analysis_interval, RATETYPE sim_time) :
    Observable(), _cells_total(cells_total),_width_total(width_total), circumf(width_total), _parameter_set(ps), _model(m), 
    _cellParams(*this, cells_total), _numNeighbors(new int[cells_total]), 
    _neighbors(new CPUGPU_TempArray<int, 6>[cells_total]), analysis_gran(analysis_interval), time_total(sim_time),
   factors_perturb(pnFactorsPert), factors_gradient(pnFactorsGrad) { }

  //DECONSTRUCTOR
  virtual ~simulation_base() {}
    
  //Virtual function all subclasses must implement
  virtual void initialize();

    /*
     * CALC_NEIGHBOR_2D
     * populates the data structure "_neighbors" with cell indices of neighbors
     * follows hexagonal adjacencies for an unfilled tube
    */
    CPUGPU_FUNC
    void calc_neighbor_2d(){
        for (int i = 0; i < _cells_total; i++) {        
	        int adjacents[6];

            /* Hexagonal Adjacencies
            0: TOP
            1: TOP-RIGHT
            2: BOTTOM-RIGHT
            3: BOTTOM
            4: TOP-LEFT
            5: BOTTOM-LEFT
            */
            if (i % 2 == 0) {
                adjacents[0] = (i - circumf + _cells_total) % _cells_total;
                adjacents[1] = (i - circumf + 1 + _cells_total) % _cells_total;
                adjacents[2] = (i + 1) % _cells_total;
                adjacents[3] = (i + circumf) % _cells_total;
                if (i % circumf == 0) {	
                    adjacents[4] = i + circumf - 1;
                    adjacents[5] = (i - 1 + _cells_total) % _cells_total;
                } else {
                    adjacents[4] = i - 1;
                    adjacents[5] = (i - circumf - 1 + _cells_total) % _cells_total;
                }
            } else {
                adjacents[0] = (i - circumf + _cells_total) % _cells_total;
                if (i % circumf == circumf - 1) {
                    adjacents[1] = i - circumf + 1;
                    adjacents[2] = (i + 1) % _cells_total;
                } else {
                    adjacents[1] = i + 1;
                    adjacents[2] = (i + circumf + 1 + _cells_total) % _cells_total;
                }
                adjacents[3] = (i + circumf) % _cells_total;
                adjacents[4] = (i + circumf - 1) % _cells_total;
                adjacents[5] = (i - 1 + _cells_total) % _cells_total;
            }
            
            if (i % circumf == 0) {
                _neighbors[i][0] = adjacents[0];
                _neighbors[i][1] = adjacents[1];
                _neighbors[i][2] = adjacents[4];
                _neighbors[i][3] = adjacents[5];
                _numNeighbors[i] = 4;
    	    }else if ((i+1) % circumf == 0) {
                _neighbors[i][0] = adjacents[0];
                _neighbors[i][1] = adjacents[1];
                _neighbors[i][2] = adjacents[2];
                _neighbors[i][3] = adjacents[3];
                _numNeighbors[i] = 4;
            } else{
                _neighbors[i][0] = adjacents[0];
                _neighbors[i][1] = adjacents[1];
                _neighbors[i][2] = adjacents[2];
                _neighbors[i][3] = adjacents[3];
                _neighbors[i][4] = adjacents[4];
                _neighbors[i][5] = adjacents[5];
                _numNeighbors[i] = 6;
            }
        }
    }
  
    //Virtual function all subclasses must implement  
    virtual void simulate() = 0;
    
    void run() final
    {
        simulate();
    }

  protected:
    void calc_max_delays();
};
#endif

