#ifndef SIM_DETERM_DETERM_HPP
#define SIM_DETERM_DETERM_HPP

#include "sim/base.hpp"
#include "core/observable.hpp"
#include "core/context.hpp"
#include "core/param_set.hpp"
#include "core/model.hpp"
#include "core/specie.hpp"
#include "sim/cell_param.hpp"
#include "core/reaction.hpp"
#include "baby_cl.hpp"

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
        IF_CUDA(__host__ __device__)
        Context(simulation_determ& sim, int cell) : _simulation(sim),_cell(cell) { }
        IF_CUDA(__host__ __device__)
        RATETYPE calculateNeighborAvg(specie_id sp, int delay = 0) const;
        IF_CUDA(__host__ __device__)
        void updateCon(const SpecieRates& rates);
        IF_CUDA(__host__ __device__)
        const SpecieRates calculateRatesOfChange();
        IF_CUDA(__host__ __device__)
        virtual RATETYPE getCon(specie_id sp) const final {
          return getCon(sp, 1);
        }
        IF_CUDA(__host__ __device__)
        RATETYPE getCon(specie_id sp, int delay) const {
            return _simulation._baby_cl[sp][1 - delay][_cell];
        }
        IF_CUDA(__host__ __device__)
        RATETYPE getCritVal(critspecie_id rcritsp) const {
            return _simulation._cellParams[rcritsp + NUM_REACTIONS + NUM_DELAY_REACTIONS][_cell];
        }
        IF_CUDA(__host__ __device__)
        RATETYPE getRate(reaction_id reaction) const {
            return _simulation._cellParams[reaction][_cell];
        }
        IF_CUDA(__host__ __device__)
        int getDelay(delay_reaction_id delay_reaction) const{
            return _simulation._cellParams[delay_reaction + NUM_REACTIONS][_cell];
        }
        IF_CUDA(__host__ __device__)
        virtual void advance() final { ++_cell; }
	IF_CUDA(__host__ __device__)
	virtual void set(int c) final {_cell = c;}
        IF_CUDA(__host__ __device__)
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

  int _j;
  int _num_history_steps; // how many steps in history are needed for this numerical method

  simulation_determ(const model& m, const param_set& ps, RATETYPE* pnFactorsPert, RATETYPE** pnFactorsGrad, int cells_total, int width_total,
                    RATETYPE step_size, RATETYPE analysis_interval, RATETYPE sim_time) :
    simulation_base(m, ps, pnFactorsPert, pnFactorsGrad, cells_total, width_total, analysis_interval, sim_time), _intDelays(*this, cells_total),
    _baby_cl(*this), _step_size(step_size), _j(0), _num_history_steps(2) { }
  virtual ~simulation_determ() {}
  void execute();
  void initialize();

    void simulate();
};
#endif
