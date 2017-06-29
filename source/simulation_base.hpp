#ifndef SIMULATION_BASE_HPP
#define SIMULATION_BASE_HPP

#include "observable.hpp"
#include "context.hpp"
#include "param_set.hpp"
#include "model.hpp"
#include "specie.hpp"
#include "cell_param.hpp"
#include "reaction.hpp"
#include "concentration_level.hpp"
#include <vector>
#include <array>
using namespace std;

/* simulation contains simulation data, partially taken from input_params and partially derived from other information
 */


typedef cell_param<NUM_REACTIONS> Rates;
typedef cell_param<NUM_DELAY_REACTIONS> Delays;
typedef cell_param<NUM_CRITICAL_SPECIES> CritValues;

class simulation_base : public Observable{
  
 public:
  // PSM stands for Presomitic Mesoderm (growth region of embryo)

  // Sizes
  int _width_total; // The width in cells of the PSM
  int circumf;
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

  // Granularities
  //int _big_gran; // The granularity in time steps with which to analyze and store data
  //int small_gran; // The granularit in time steps with which to simulate data

  // Neighbors and boundaries
  //array2D<int> neighbors; // An array of neighbor indices for each cell position used in 2D simulations (2-cell and 1D calculate these on the fly)
  //int active_start; // The start of the active portion of the PSM
  //int active_end; // The end of the active portion of the PSM

  // PSM section and section-specific times
  //int section; // Posterior or anterior (sec_post or sec_ant)
  //int time_start; // The start time (in time steps) of the current simulation
  //int time_end; // The end time (in time steps) of the current simulation

  // Mutants and condition scores
  //int num_active_mutants; // The number of mutants to simulate for each parameter set
  //double max_scores[NUM_SECTIONS]; // The maximum score possible for all mutants for each testing section
  //double max_score_all; // The maximum score possible for all mutants for all testing sections

  const param_set& _parameter_set;
  const model& _model;
  Rates _rates;
  Delays _delays;
  CritValues _critValues;
  //Context<double> _contexts;
  //CPUGPU_TempArray<int,NUM_SPECIES> _baby_j;
  //int* _delay_size;
  //int* _time_prev;
  vector<vector<int> > _neighbors;
  //double* _sets;
  //int _NEIGHBORS_2D;
  RATETYPE max_delays[NUM_SPECIES];  // The maximum number of time steps that each specie might be accessed in the past
    

    
  simulation_base(const model& m, const param_set& ps, int cells_total, int width_total, RATETYPE analysis_interval, RATETYPE sim_time) :
    _cells_total(cells_total),_width_total(width_total), circumf(width_total), _parameter_set(ps), _model(m), 
    _rates(*this, cells_total), _delays(*this, cells_total), _critValues(*this, cells_total), 
    analysis_gran(analysis_interval), time_total(sim_time) { }

  virtual ~simulation_base() {}
    
  virtual void initialize();
#ifdef __CUDACC__
    __host__ __device__
#endif
    void calc_neighbor_2d(){
        for (int i = 0; i < _cells_total; i++) {
            
            _neighbors[i].clear();

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
            
            _neighbors[i].push_back(i);
            if (i % circumf == 0) {
                _neighbors[i].push_back(adjacents[0]);
                _neighbors[i].push_back(adjacents[3]);
                _neighbors[i].push_back(adjacents[4]);
		_neighbors[i].push_back(adjacents[5]);
    	    }else if ((i+1) % circumf == 0) {
                _neighbors[i].push_back(adjacents[0]);
                _neighbors[i].push_back(adjacents[1]);
                _neighbors[i].push_back(adjacents[2]);
                _neighbors[i].push_back(adjacents[3]);
            } else{
                _neighbors[i].push_back(adjacents[0]);
                _neighbors[i].push_back(adjacents[1]);
                _neighbors[i].push_back(adjacents[2]);
                _neighbors[i].push_back(adjacents[3]);
                _neighbors[i].push_back(adjacents[4]);
                _neighbors[i].push_back(adjacents[5]);
            }
        }
    }
    
    virtual void simulate() = 0;
 protected:
  void calc_max_delays();
};
#endif

