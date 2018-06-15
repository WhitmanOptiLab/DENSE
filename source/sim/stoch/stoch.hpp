#ifndef SIM_STOCH_HPP
#define SIM_STOCH_HPP

#include "sim/base.hpp"
#include "core/observable.hpp"
#include "core/context.hpp"
#include "core/parameter_set.hpp"
#include "core/model.hpp"
#include "core/specie.hpp"
#include "sim/cell_param.hpp"
#include "core/reaction.hpp"
#include <vector>
#include <set>
#include <random>

/*
 * STOCHASTIC SIMULATOR:
 * superclasses: simulation_base, Observable
 * uses Gillespie's tau leaping algorithm
 * uses Barrio's delay SSA
*/
class simulation_stoch : public Simulation {

 private:

    //"event" represents a delayed reaction scheduled to fire later
    struct event{
      RATETYPE time;
	    int cell;
      reaction_id rxn;
	    RATETYPE getTime() const{return time;}
	    bool operator<(event const &b) const {return time < b.time;}
    };

    //"event_schedule" is a set ordered by time of delay reactions that will fire
    std::multiset<event> event_schedule;

    //"concs" stores current concentration levels for every species in every cell
    std::vector<std::vector<int> > concs;
    //"t" is current simulation time
    double littleT;
    //"propensities" stores probability of each rxn firing, calculated from active rates
    std::vector<std::vector<RATETYPE> > propensities;
    //for each rxn, stores intracellular reactions whose rates are affected by a firing of that rxn
    std::vector<reaction_id> propensity_network[NUM_REACTIONS];
    //for each rxn, stores intercellular reactions whose rates are affected by a firing of that rxn
    std::vector<reaction_id> neighbor_propensity_network[NUM_REACTIONS];
    //random number generator
    std::default_random_engine generator;

    RATETYPE generateTau();
    RATETYPE getSoonestDelay();
    void executeDelayRXN();
    RATETYPE getRandVariable();
    void tauLeap(RATETYPE tau);
    void initPropensityNetwork();
    void generateRXNTaus(RATETYPE tau);
    void fireOrSchedule(int c, reaction_id rid);
    void initPropensities();

    public:

    /*
     * ContextStoch:
     * iterator for observers to access conc levels with
    */
    class ContextStoch : public ContextBase {
        //FIXME - want to make this private at some point
      private:
        simulation_stoch& _simulation;
        double _avg;
        int _cell;

      public:
        typedef CPUGPU_TempArray<RATETYPE, NUM_SPECIES> SpecieRates;
        IF_CUDA(__host__ __device__)
        ContextStoch(simulation_stoch& sim, int cell) : _simulation(sim), _cell(cell) { }
        IF_CUDA(__host__ __device__)
        RATETYPE calculateNeighborAvg(specie_id sp, int delay) const;
        IF_CUDA(__host__ __device__)
        void updateCon(specie_id sid,int delta){
	      if (_simulation.concs[_cell][sid]+delta < 0){
              _simulation.concs[_cell][sid] = 0;
          }
          else{
              _simulation.concs[_cell][sid]+=delta;
          }
	    }
        IF_CUDA(__host__ __device__)
        void updatePropensities(reaction_id rid);
	    IF_CUDA(__host__ __device__)
	    RATETYPE getTotalPropensity();
	    IF_CUDA(__host__ __device__)
	    int chooseReaction(RATETYPE propensity_portion);
        IF_CUDA(__host__ __device__)
        virtual RATETYPE getCon(specie_id sp) const final {
          return _simulation.concs[_cell][sp];
        }
	    RATETYPE getCon(specie_id sp, int delay) const {
	      return getCon(sp);
	    }
        IF_CUDA(__host__ __device__)
        RATETYPE getCritVal(critspecie_id rcritsp) const {
            return _simulation._cellParams[NUM_REACTIONS+NUM_DELAY_REACTIONS+rcritsp][_cell];
        }
        IF_CUDA(__host__ __device__)
        RATETYPE getRate(reaction_id reaction) const {
            return _simulation._cellParams[reaction][_cell];
        }
        IF_CUDA(__host__ __device__)
        RATETYPE getDelay(delay_reaction_id delay_reaction) const{
            return _simulation._cellParams[NUM_REACTIONS+delay_reaction][_cell];
        }
        IF_CUDA(__host__ __device__)
        virtual void advance() final { ++_cell; }
	    IF_CUDA(__host__ __device__)
	    virtual void set(int c) final {_cell = c;}
        IF_CUDA(__host__ __device__)
        virtual bool isValid() const final { return _cell >= 0 && _cell < _simulation._cells_total; }
    };

  private:
    void fireReaction(ContextStoch *c, const reaction_id rid);

  public:
    /*
     * Constructor:
     * calls simulation base constructor
     * initializes fields "t" and "generator"
    */
    simulation_stoch(const model& m, const Parameter_Set& ps, RATETYPE* pnFactorsPert, RATETYPE** pnFactorsGrad, int cells_total, int width_total,
                    RATETYPE analysis_interval, RATETYPE sim_time, int seed):
        Simulation(m, ps, pnFactorsPert, pnFactorsGrad, cells_total, width_total, analysis_interval, sim_time),
        generator(std::default_random_engine(seed)){}

    //Deconstructor
    virtual ~simulation_stoch() {}

    void initialize();
    void simulate();
};
#endif
