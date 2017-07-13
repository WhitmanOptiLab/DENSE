#ifndef SIM_STOCH_HPP
#define SIM_STOCH_HPP

#include "sim/base.hpp"
#include "core/observable.hpp"
#include "core/context.hpp"
#include "core/param_set.hpp"
#include "core/model.hpp"
#include "core/specie.hpp"
#include "sim/cell_param.hpp"
#include "core/reaction.hpp"
#include <vector>
#include <set>
#include <random>

using namespace std;


typedef cell_param<NUM_DELAY_REACTIONS, int> IntDelays;

/*
 * STOCHASTIC SIMULATOR:
 * superclasses: simulation_base, Observable
 * uses Gillespie's tau leaping algorithm
 * uses Barrio's delay SSA
*/
class simulation_stoch : public simulation_base {

 private:

    //"event" represents a delayed reaction scheduled to fire later
    struct event{
        RATETYPE time;
	    RATETYPE getTime() const{return time;}
	    int cell;
        reaction_id rxn;
	    bool operator<(event const &b) const {return getTime() < b.getTime();}
    };

    //"event_schedule" is a set ordered by time of delay reactions that will fire
    multiset<event> event_schedule;

    //"concs" stores current concentration levels for every species in every cell
    vector<vector<int> > concs;
    //"t" is current simulation time
    long double t, littleT;
    //"propensities" stores probability of each rxn firing, calculated from active rates
    vector<vector<RATETYPE> > propensities;
    //for each rxn, stores intracellular reactions whose rates are affected by a firing of that rxn
    vector<reaction_id> propensity_network[NUM_REACTIONS];
    //for each rxn, stores intercellular reactions whose rates are affected by a firing of that rxn
    vector<reaction_id> neighbor_propensity_network[NUM_REACTIONS];
    //random number generator
    default_random_engine generator;

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
    /*
     * Constructor:
     * calls simulation base constructor
     * initializes fields "t" and "generator"
    */
    simulation_stoch(const model& m, const param_set& ps, RATETYPE* pnFactorsPert, RATETYPE** pnFactorsGrad, int cells_total, int width_total,
                    RATETYPE analysis_interval, RATETYPE sim_time, int seed):
        simulation_base(m, ps, pnFactorsPert, pnFactorsGrad, cells_total, width_total, analysis_interval, sim_time),
        generator(default_random_engine(seed)), t(0){}

    //Deconstructor
    virtual ~simulation_stoch() {}

    void initialize();
    void simulate();
};
#endif
