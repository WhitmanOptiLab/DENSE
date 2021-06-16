#ifndef SIM_LOG_DIRECT_HPP
#define SIM_LOG_DIRECT_HPP

#include "sim/base.hpp"
#include "core/parameter_set.hpp"
#include "core/model.hpp"
#include "core/specie.hpp"
#include "sim/cell_param.hpp"
#include "core/reaction.hpp"
#include "datastructures/modifiable_heap_random_selector.hpp"
#include "Sim_Builder.hpp"
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

    Log_Direct_Simulation(const Parameter_Set& ps, NGraph::Graph adj_graph, std::vector<int> conc, Real* pnFactorsPert, Real** pnFactorsGrad, int seed)
    : Simulation(ps, adj_graph, pnFactorsPert, pnFactorsGrad)
    , concs(cell_count(), conc)
    , propensities(cell_count()*NUM_REACTIONS)
    , generator{seed} {
      initPropensityNetwork();
      initPropensities();
    }
  
    Real get_total_propensity() const {return propensities.total_weight();}

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
                propensities.update_weight(encode(cell_,name), \
                    dense::model::reaction_##name.active_rate(Context(*this, cell_))); \
            } \
        } \
        for (std::size_t r=0; r< neighbor_propensity_network[rid].size(); r++) { \
            if (name == neighbor_propensity_network[rid][r]) { \
                for (dense::Natural n=0; n < neighbor_count_by_cell_[cell_]; n++) { \
                    int n_cell = neighbors_by_cell_[cell_][n]; \
                    Context neighbor(*this, n_cell); \
                    propensities.update_weight(encode(n_cell, name), \
                        dense::model::reaction_##name.active_rate(neighbor)); \
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
using stochastic::Log_Direct_Simulation;

template<>
class Sim_Builder <Log_Direct_Simulation> : public Sim_Builder_Stoch {
  using This = Sim_Builder<Log_Direct_Simulation>;

  public:
  This& operator= (This&&);
  Sim_Builder (This const&) = default;
  Sim_Builder(Real* pf, Real** gf, NGraph::Graph adj_graph, int argc, char* argv[]) : 
    Sim_Builder_Stoch(pf, gf, adj_graph, argc, argv) {}

  std::vector<Log_Direct_Simulation> get_simulations(std::vector<Parameter_Set> param_sets){
    std::vector<Log_Direct_Simulation> simulations;
    for (auto& parameter_set : param_sets) {
      simulations.emplace_back(std::move(parameter_set), adjacency_graph, conc, perturbation_factors, gradient_factors, seed);
    }
    return simulations;
  };
};

}

#endif
