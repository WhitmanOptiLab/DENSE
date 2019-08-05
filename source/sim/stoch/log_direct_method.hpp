#ifndef SIM_LOG_DIRECT_HPP
#define SIM_LOG_DIRECT_HPP

#include "sim/base.hpp"
#include "core/parameter_set.hpp"
#include "core/model.hpp"
#include "core/specie.hpp"
#include "sim/cell_param.hpp"
#include "core/reaction.hpp"
#include "modifiable_heap_random_selector.hpp"
#include <vector>
#include <set>
#include <queue>
#include <random>

namespace dense {
namespace stochastic {

/*
 * STOCHASTIC SIMULATOR:
 * superclasses: simulation_base, Observable
 * uses Gillespie's tau leaping algorithm
 * uses Barrio's delay SSA
*/
class Log_Direct_Simulation : public Simulation {

public:

    using Context = dense::Context<Log_Direct_Simulation>;

 private:

    //"event" represents a delayed reaction scheduled to fire later
    struct event {
      Minutes time;
	    Natural cell;
      reaction_id rxn;
	    friend bool operator<(event const& a, event const &b) { return a.time < b.time;}
	    friend bool operator>(event const& a, event const& b) { return b < a; }
    };

    /*
    //"event_schedule" is a set ordered by time of delay reactions that will fire
    std::priority_queue<event, std::vector<event>, std::greater<event>> event_schedule;
    */
    
    //"concs" stores current concentration levels for every species in every cell
    std::vector<std::vector<int> > concs;
    //"propensities" stores probability of each rxn firing, calculated from active rates, and 
    //  enables weighted random selection from among its elements
    fast_random_selector<int> propensities;
    //for each rxn, stores intracellular reactions whose rates are affected by a firing of that rxn
    std::vector<reaction_id> propensity_network[NUM_REACTIONS];
    //for each rxn, stores intercellular reactions whose rates are affected by a firing of that rxn
    std::vector<reaction_id> neighbor_propensity_network[NUM_REACTIONS];
    //random number generator
    std::default_random_engine generator = std::default_random_engine{ std::random_device()() };

    Real total_propensity_ = {};
    static std::uniform_real_distribution<Real> distribution_;

    Minutes generateTau();
    Minutes getSoonestDelay() const;
    void executeDelayRXN();
    Real getRandVariable();
    void tauLeap();
    void initPropensityNetwork();
    void generateRXNTaus(Real tau);
    void fireOrSchedule(int c, reaction_id rid);
    void initPropensities();

    public:

    /*
     * ContextStoch:
     * iterator for observers to access conc levels with
    */
    using SpecieRates = CUDA_Array<Real, NUM_SPECIES>;

  private:
    void fireReaction(dense::Natural cell, const reaction_id rid);

  public:
    /*
     * Constructor:
     * calls simulation base constructor
     * initializes fields "t" and "generator"
    */

    Sorting_Direct_Simulation(const Parameter_Set& ps, NGraph::Graph adj_graph, std::vector<int> conc, Real* pnFactorsPert, Real** pnFactorsGrad, int seed)
    : Simulation(ps, adj_graph, pnFactorsPert, pnFactorsGrad)
    , concs(cell_count(), conc)
    , propensities(cell_count()*NUM_REACTIONS)
    , generator{seed} {
      initPropensityNetwork();
      initPropensities();
    }
  
    std::vector<Real> get_perf(){
      return Simulation::get_performance();
    }


    Real get_concentration (dense::Natural cell, specie_id species) const {
      return concs.at(cell).at(species);
    }

    Real get_concentration (dense::Natural cell, specie_id species, dense::Natural delay) const {
      (void)delay;
      return get_concentration(cell, species);
    }

    void update_concentration (dense::Natural cell_, specie_id sid, int delta) {
      auto& concentration = concs[cell_][sid];
      concentration = std::max(concentration + delta, 0);
    }


    /*
     * CHOOSEREACTION
     * randomly chooses a reaction biased by their propensities
     * arg "propensity_portion": the propensity sum times a random variable between 0.0 and 1.0
     * return "j": the index of the reaction chosen in RSO.
    */
    CUDA_AGNOSTIC
    __attribute_noinline__ int choose_reaction() {
      return propensities(generator);
    }

    /*
     * UPDATEPROPENSITIES
     * recalculates the propensities of reactions affected by the firing of "rid"
     * arg "rid": the reaction that fired
    */
    CUDA_AGNOSTIC
    __attribute_noinline__ void update_propensities(dense::Natural cell_, reaction_id rid) {
        #define REACTION(name) \
        for (std::size_t i=0; i< propensity_network[rid].size(); i++) { \
            if ( name == propensity_network[rid][i] ) { \
                auto& p = propensities.update(encode(cell_,name), \
                    dense::model::reaction_##name.active_rate(Context(*this, cell_)); \
            } \
        } \
        for (std::size_t r=0; r< neighbor_propensity_network[rid].size(); r++) { \
            if (name == neighbor_propensity_network[rid][r]) { \
                for (dense::Natural n=0; n < neighbor_count_by_cell_[cell_]; n++) { \
                    int n_cell = neighbors_by_cell_[cell_][n]; \
                    Context neighbor(*this, n_cell); \
                    auto& p = propensities.update(encode(n_cell, name), \
                        dense::model::reaction_##name.active_rate(neighbor); \
                } \
            } \
        }
        #include "reactions_list.hpp"
        #undef REACTION
    }


  /*
   * CALCULATENEIGHBORAVG
   * arg "sp": the specie to average from the surrounding cells
   * arg "delay": unused, but used in deterministic context. Kept for polymorphism
   * returns "avg": average concentration of specie in current and neighboring cells
  */
  Real calculate_neighbor_average (dense::Natural cell, specie_id species, dense::Natural delay) const {
    (void)delay;
    Real sum = 0;
    for (dense::Natural i = 0; i < neighbor_count_by_cell_[cell]; ++i) {
      sum += concs[neighbors_by_cell_[cell][i]][species];
    }
    return sum / neighbor_count_by_cell_[cell];
  }

  Minutes age_by(Minutes duration);

  private:
    int encode(Natural cell, reaction_id reaction){
			Natural rxn_id = static_cast<Natural>(reaction);
			return (cell*NUM_REACTIONS)+rxn_id;
		}

    Minutes time_until_next_event () const;

};

}
}

#endif
