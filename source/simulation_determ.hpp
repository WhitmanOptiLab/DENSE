#ifndef SIMULATION_DETERM_HPP
#define SIMULATION_DETERM_HPP

#include "simulation_base.hpp"
#include "observable.hpp"
#include "context.hpp"
#include "param_set.hpp"
#include "model.hpp"
#include "specie.hpp"
#include "cell_param.hpp"
#include "reaction.hpp"
#include "baby_cl.hpp"

using namespace std;

/* simulation contains simulation data, partially taken from input_params and partially derived from other information
 */

typedef cell_param<NUM_DELAY_REACTIONS, int> IntDelays;

class simulation_determ : public simulation_base {
  
 public:
    class Context : public ContextBase {
        //FIXME - want to make this private at some point
      protected:
        int _cell;
        simulation_determ& _simulation;
        double _avg;

      public:
        typedef CPUGPU_TempArray<RATETYPE, NUM_SPECIES> SpecieRates;
        CPUGPU_FUNC
        Context(simulation_determ& sim, int cell) : _simulation(sim),_cell(cell) { }
        CPUGPU_FUNC
        RATETYPE calculateNeighborAvg(specie_id sp, int delay = 0) const;
        CPUGPU_FUNC
        void updateCon(const SpecieRates& rates);
        CPUGPU_FUNC
        const SpecieRates calculateRatesOfChange();
        CPUGPU_FUNC
        virtual RATETYPE getCon(specie_id sp) const final {
          return getCon(sp, 1);
        }
        CPUGPU_FUNC
        RATETYPE getCon(specie_id sp, int delay) const {
            return _simulation._baby_cl[sp][1 - delay][_cell];
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
        int getDelay(delay_reaction_id delay_reaction) const{
            return _simulation._intDelays[delay_reaction][_cell];
        }
        CPUGPU_FUNC
        virtual void advance() final { ++_cell; }
	CPUGPU_FUNC
	virtual void set(int c) final {_cell = c;}
        CPUGPU_FUNC
        virtual bool isValid() const final { return _cell >= 0 && _cell < _simulation._cells_total; }
    };
  // PSM stands for Presomitic Mesoderm (growth region of embryo)
  IntDelays _intDelays;

  baby_cl _baby_cl;

  // Sizes
  RATETYPE _step_size; // The step size in minutes
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
  //int time_baby; // Time 0 for baby_cl at the end of a simulation

  // Mutants and condition scores
  //int num_active_mutants; // The number of mutants to simulate for each parameter set
  //double max_scores[NUM_SECTIONS]; // The maximum score possible for all mutants for each testing section
  //double max_score_all; // The maximum score possible for all mutants for all testing sections

  //Context<double> _contexts;
  //CPUGPU_TempArray<int,NUM_SPECIES> _baby_j;
  //int* _delay_size;
  //int* _time_prev;
  int _j;
  //double* _sets;
  //int _NEIGHBORS_2D;
  int _num_history_steps; // how many steps in history are needed for this numerical method
  //int* _relatedReactions[NUM_SPECIES];
    
  simulation_determ(const model& m, const param_set& ps, int cells_total, int width_total, 
                    RATETYPE step_size, RATETYPE analysis_interval, RATETYPE sim_time) :
    simulation_base(m, ps, cells_total, width_total, analysis_interval, sim_time), _intDelays(*this, cells_total),
    _baby_cl(*this), _step_size(step_size), _j(0), _num_history_steps(2) { }
  virtual ~simulation_determ() {}
  void execute();
  //bool any_less_than_0(baby_cl& baby_cl, int* times);
  //bool concentrations_too_high (baby_cl& baby_cl, int* time, double max_con_thresh);
#if 0
#ifdef __CUDACC__
  __host__ __device__
#endif
    void calculate_delay_indices(baby_cl& baby_cl, int* baby_time, int time, int cell_index, Rates& rs, int old_cells_mrna[], int old_cells_protein[]){
        for (int l = 0; l < NUM_SPECIES; l++) {
            old_cells_mrna[l] = cell_index;
            old_cells_protein[l] = cell_index;
        }
    }
#endif
    
  void initialize();
    
    void simulate();
 protected:
  void calc_max_delays();
};
#endif

