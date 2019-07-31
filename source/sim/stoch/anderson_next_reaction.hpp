#ifndef SIM_STOCH_ANDERSON_NEXT_REACTION_SIMULATION_HPP
#define SIM_STOCH_ANDERSON_NEXT_REACTION_SIMULATION_HPP

#include "sim/base.hpp"
#include "core/parameter_set.hpp"
#include "core/model.hpp"
#include "core/specie.hpp"
#include "sim/cell_param.hpp"
#include "core/reaction.hpp"
#include <vector>
#include <set>
#include <queue>
#include <random>
#include <algorithm>

namespace dense {
namespace stochastic {

/*
 * Next_Reaction_Simulation
 * uses the Next Reaction Algorithm
*/
class Anderson_Next_Reaction_Simulation : public Simulation {

public:

    using Context = dense::Context<Anderson_Next_Reaction_Simulation>;

    using event_id = Natural;
 private:

    //"event" represents a delayed reaction scheduled to fire later
    struct event {
      Minutes time;
      Natural cell;
      reaction_id reaction;
      friend bool operator<(event const& a, event const &b) { return a.time < b.time;}
      friend bool operator>(event const& a, event const& b) { return b < a; }
    };
    
    //"reaction_schedule" is a set ordered by time of delay reactions that will fire
    //indexed_priority_queue<event_id, Minutes> reaction_schedule;

    //"concs" stores current concentration levels for every species in every cell
    std::vector<std::vector<int>> concs;
    //"propensities" stores probability of each rxn firing, calculated from active rates
    std::vector<std::vector<Real>> propensities;
    //P_k is the first firing time of Y_k 
    std::vector<std::vector<Real>> P;
    //T stores internal time for each k
    std::vector<std::vector<Real>> T;
    //"rand_num" stores random number for each k
    std::vector<std::vector<Real>> rand_num;
    std::vector<std::vector<Real>> delta_tk;
    //for each rxn, stores intracellular reactions whose rates are affected by a firing of that rxn
    std::vector<reaction_id> propensity_network[NUM_REACTIONS];
    //for each rxn, stores intercellular reactions whose rates are affected by a firing of that rxn
    std::vector<reaction_id> neighbor_propensity_network[NUM_REACTIONS];
    //random number generator
    std::default_random_engine generator;

    Real total_propensity_ = {};
    static std::uniform_real_distribution<Real> distribution_;

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
    dense::Natural c_miu;
    reaction_id r_miu;
    Real delta_minimum;

  private:
    void fireReaction(dense::Natural cell, const reaction_id rid);
    

  public:
    /*
     * Constructor:
     * calls simulation base constructor
     * initializes fields "t" and "generator"
    */
    Anderson_Next_Reaction_Simulation (const Parameter_Set& ps, Real* pnFactorsPert, Real** pnFactorsGrad, int cell_count, int width_total, int seed)
    : Simulation(ps, cell_count, width_total, pnFactorsPert, pnFactorsGrad)
    , concs(cell_count, std::vector<int>(NUM_SPECIES, 0))
    , propensities(cell_count)
    , generator{seed} {
      // 1. initialize
      initPropensityNetwork();
      // 2. calculate the propensity function a_k for all k
      initPropensities();
      // generate random number r_k, init P and get the first delta_tk  
      Real& delta_min = delta_minimum;
      delta_min = std::numeric_limits<Real>::infinity();
      dense::Natural& cell_miu = c_miu;
      reaction_id& reaction_miu = r_miu;
      for (dense::Natural c = 0; c < cell_count; ++c) {
        std::vector<Real> temp_P;
        for (int i = 0; i < NUM_REACTIONS; ++i) {
          Real r_num = getRandVariable();
          Real p = log(1/r_num);
          Real tk = p/propensities[c][i];
          temp_P.push_back(p);
          if (delta_min > tk){
            delta_min = tk;
            cell_miu = c;
            reaction_miu = static_cast<reaction_id>(i);
          }
        }
        P.push_back(temp_P);
      }
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
   * GETTOTALPROPENSITY
   * sums the propensities of every reaction in every cell
   * called by "generateTau" in simulation_stoch.cpp
   * return "sum": the propensity sum
  */
   // Todo: store this as a cached variable and change it as propensities change;
   // sum += new_value - old_value;
    __attribute_noinline__ Real get_total_propensity() const {
      Real sum = total_propensity_; // 0.0;
      /*for (dense::Natural c = 0; c < _cells_total; ++c) {
        for (int r=0; r<NUM_REACTIONS; r++) {
          sum += propensities[c][r];
        }
      }*/
      return sum;
    }


    /*
     * UPDATEPROPENSITIES
     * recalculates the propensities of reactions affected by the firing of "rid"
     * arg "rid": the reaction that fired
    */
    CUDA_AGNOSTIC
    __attribute_noinline__ void update_propensities(dense::Natural cell_, reaction_id rid) {
        #define REACTION(name)\
        for (std::size_t i=0; i< propensity_network[rid].size(); i++) { \
            if (name == propensity_network[rid][i]) { /* alpha == mu */\
              /* 5.a. update a_alpha (propensity) */\
              auto& a = propensities[cell_][name]; \
              auto new_a = std::max(dense::model::reaction_##name.active_rate(Context(*this, cell_)), Real{0}); \
              a = new_a;\
            } \
        } \
        for (std::size_t r=0; r< neighbor_propensity_network[rid].size(); r++) { \
            if (name == neighbor_propensity_network[rid][r]) { \
                for (dense::Natural n=0; n < neighbor_count_by_cell_[cell_]; n++) { \
                    int n_cell = neighbors_by_cell_[cell_][n]; \
                    Context neighbor(*this, n_cell); \
                    auto& a = propensities[n_cell][name];\
                    auto new_a = dense::model::reaction_##name.active_rate(neighbor); \
                    a = new_a;\
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

    Minutes time_until_next_event () const;
	
		event_id encode(Natural cell, reaction_id reaction){
			Natural rxn_id = static_cast<Natural>(reaction);
			return (cell*NUM_REACTIONS)+rxn_id;
		}
		std::pair<Natural, reaction_id> decode(event_id e){
			reaction_id rxn_id = static_cast<reaction_id>(e % NUM_REACTIONS);
			Natural c = e / NUM_REACTIONS;
			return std::make_pair(c,rxn_id);
		}
	
    void initTau();

  
};


}
}
#endif