#ifndef SIMULATION_STOCH_HPP
#define SIMULATION_STOCH_HPP

#include "simulation_base.hpp"
#include "observable.hpp"
#include "context.hpp"
#include "param_set.hpp"
#include "model.hpp"
#include "specie.hpp"
#include "cell_param.hpp"
#include "reaction.hpp"
#include "concentration_level.hpp"
#include "baby_cl.hpp"
#include <vector>
#include <array>
#include <set>
#include <random>

using namespace std;

/* simulation contains simulation data, partially taken from input_params and partially derived from other information
 */

typedef cell_param<NUM_DELAY_REACTIONS, int> IntDelays;

class simulation_stoch : public simulation_base {

 private:

    struct event{
        RATETYPE time;
	RATETYPE getTime() const{return time;}
	int cell;
        reaction_id rxn;
	bool operator<(event const &b) const {return getTime() < b.getTime();}
    };
    multiset<event> event_schedule;
    vector<vector<int> > concs;
    long double t, littleT;
    vector<vector<RATETYPE> > propensities;
    vector<reaction_id> propensity_network[NUM_REACTIONS];
    vector<reaction_id> neighbor_propensity_network[NUM_REACTIONS];

    RATETYPE generateTau();
    RATETYPE getSoonestDelay();
    void executeDelayRXN();
    RATETYPE getRandVariable();
    void tauLeap(RATETYPE tau);
    void initPropensityNetwork();
    void generateRXNTaus(RATETYPE tau);
    void fireOrSchedule(int c, reaction_id rid);
    void initPropensities();
    default_random_engine generator;

    public:

    class ContextStoch : public ContextBase {
        //FIXME - want to make this private at some point
      private:
        int _cell;
        simulation_stoch& _simulation;
        double _avg;

      public:
        typedef CPUGPU_TempArray<RATETYPE, NUM_SPECIES> SpecieRates;	
        CPUGPU_FUNC
        ContextStoch(simulation_stoch& sim, int cell) : _simulation(sim),_cell(cell) { }
        CPUGPU_FUNC
        RATETYPE calculateNeighborAvg(specie_id sp, int delay) const;
        CPUGPU_FUNC
        void updateCon(specie_id sid,int delta){
	      if (_simulation.concs[_cell][sid]+delta < 0){
              _simulation.concs[_cell][sid] = 0;
          }
          else{
              _simulation.concs[_cell][sid]+=delta;
          }
	    }
        CPUGPU_FUNC
        void updatePropensities(reaction_id rid);
	    CPUGPU_FUNC
	    RATETYPE getTotalPropensity();
	    CPUGPU_FUNC
	    int chooseReaction(RATETYPE propensity_portion);
        CPUGPU_FUNC
        virtual RATETYPE getCon(specie_id sp) const final {
          return _simulation.concs[_cell][sp];
        }
	    RATETYPE getCon(specie_id sp, int delay) const {
	      return getCon(sp);
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
            return _simulation._delays[delay_reaction][_cell];
        }
        CPUGPU_FUNC
        virtual void advance() final { ++_cell; }
	    CPUGPU_FUNC
	    virtual void reset() final {_cell = 0;}
        CPUGPU_FUNC
        virtual bool isValid() const final { return _cell >= 0 && _cell < _simulation._cells_total; }
    };

  private:
    void fireReaction(ContextStoch *c, const reaction_id rid);

  public:
  // PSM stands for Presomitic Mesoderm (growth region of embryo)

  // Sizes
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
  //double* _sets;
  //int _NEIGHBORS_2D;
  //int* _relatedReactions[NUM_SPECIES];
    
  simulation_stoch(const model& m, const param_set& ps, int cells_total, int width_total, RATETYPE analysis_interval, RATETYPE sim_time, unsigned seed): generator(default_random_engine(seed)),
    simulation_base(m, ps, cells_total, width_total, analysis_interval, sim_time),t(0){ }

  virtual ~simulation_stoch() {}
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
