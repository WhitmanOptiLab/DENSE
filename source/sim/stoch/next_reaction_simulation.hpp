#ifndef SIM_STOCH_NEXT_REACTION_SIMULATION_HPP
#define SIM_STOCH_NEXT_REACTION_SIMULATION_HPP

#include "sim/base.hpp"
#include "core/parameter_set.hpp"
#include "core/model.hpp"
#include "core/specie.hpp"
#include "sim/cell_param.hpp"
#include "core/reaction.hpp"
#include "indexed_priority_queue.hpp"
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
class Next_Reaction_Simulation : public Simulation {

public:

    using Context = dense::Context<Next_Reaction_Simulation>;

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
    indexed_priority_queue<event_id, Minutes> reaction_schedule;
    //indexed_priority_queue<event_id, Minutes> reaction_schedule;

    //"concs" stores current concentration levels for every species in every cell
    std::vector<std::vector<int> > concs;
    //"propensities" stores probability of each rxn firing, calculated from active rates
    std::vector<std::vector<Real> > propensities;
    //for each rxn, stores intracellular reactions whose rates are affected by a firing of that rxn
    std::vector<reaction_id> propensity_network[NUM_REACTIONS];
    //for each rxn, stores intercellular reactions whose rates are affected by a firing of that rxn
    std::vector<reaction_id> neighbor_propensity_network[NUM_REACTIONS];
    //random number generator
    std::default_random_engine generator = std::default_random_engine{ std::random_device()() };

    Real total_propensity_ = {};
    static std::uniform_real_distribution<Real> distribution_;

    Minutes generateTau(Real);
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
    Next_Reaction_Simulation(const Parameter_Set& ps, Real* pnFactorsPert, Real** pnFactorsGrad, int cell_count, int width_total, int seed)
    : Simulation(ps, cell_count, width_total, pnFactorsPert, pnFactorsGrad)
    , concs(cell_count, std::vector<int>(NUM_SPECIES, 0))
    , propensities(cell_count)
    , generator{seed}
	  , reaction_schedule(NUM_REACTIONS * cell_count) {
      // 1.a. generate the dependency graph
      initPropensityNetwork();
      // 1.b. calculate the propensity function a_i for all i
      initPropensities();
      // 1.c. for each i, generate a putative time, tau_i according to
      //      an exponential distribution with parameter a_i;
      // 1.d. store the values in an indexed priority queue P;
      for (dense::Natural c = 0; c < cell_count; ++c) {
        for (int i = 0; i < NUM_REACTIONS; ++i) {
          auto r = static_cast<reaction_id>(i);
          auto eid = encode(c, r);
          auto tau = generateTau(propensities[c][r]);
          reaction_schedule.emplace(eid, tau);
        }
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
     * CHOOSEREACTION
     * Chooses reaction with smallest tau
		 * returns j; the reaction id
    */
    CUDA_AGNOSTIC
    __attribute_noinline__ std::pair<Natural, reaction_id> choose_reaction() {
    		auto next_reaction_encoded_id = reaction_schedule.top().first;
				auto decoded_event_id = decode(next_reaction_encoded_id);
				return decoded_event_id;
    }

    /*
     * UPDATEPROPENSITIES
     * recalculates the propensities of reactions affected by the firing of "rid"
     * arg "rid": the reaction that fired
    */
    CUDA_AGNOSTIC
    __attribute_noinline__ void update_propensities_and_taus(dense::Natural cell_, reaction_id rid) {
        #define REACTION(name)\
        for (std::size_t i=0; i< propensity_network[rid].size(); i++) { \
            if (name == rid) { /* alpha == mu */\
              /* 5.a. update a_alpha (propensity) */\
              auto& a = propensities[cell_][name];\
              auto new_a = std::max(dense::model::reaction_##name.active_rate(Context(*this, cell_)), Real{0}); \
              a = new_a;\
              /* 5.c. generate a random number according to an exponential\
                      distribution with parameter a_mu */\
              auto p = generateTau(new_a);\
              /* Set tau_alpha <- p + t */\
              auto tau_alpha = p + age();\
              /* 5.d. replace the old tau_alpha in P with the new value */\
              reaction_schedule.emplace(encode(cell_, name), tau_alpha);\
            } else if ( name == propensity_network[rid][i] ) { /* alpha != mu */\
              /* 5.a. update a_alpha (propensity) */\
              auto& a = propensities[cell_][name];\
              auto old_a = a;\
              auto new_a = std::max(dense::model::reaction_##name.active_rate(Context(*this, cell_)), Real{0});\
              a = new_a;\
              /* 5.b. set tau_alpha <- (a_alpha_old / a_alpha_new)(tau_alpha - t) + t */\
              auto tau_alpha = reaction_schedule.at(encode(cell_, name));\
              tau_alpha = (old_a / new_a)*(tau_alpha - age()) + age();\
              /* 5.d. replace the old tau_alpha in P with the new value */\
              reaction_schedule.emplace(encode(cell_, name), tau_alpha);\
            } \
        } \
        for (std::size_t r=0; r< neighbor_propensity_network[rid].size(); r++) { \
            if (name == neighbor_propensity_network[rid][r]) { \
                for (dense::Natural n=0; n < neighbor_count_by_cell_[cell_]; n++) { \
                    int n_cell = neighbors_by_cell_[cell_][n]; /* alpha != mu */\
                    /* 5.a. update a_alpha (propensity) */\
                    auto& a = propensities[n_cell][name];\
                    auto old_a = a;\
                    auto new_a = std::max(dense::model::reaction_##name.active_rate(Context(*this, n_cell)), Real{0}); \
                    a = new_a;\
                    /* 5.b. set tau_alpha <- (a_alpha_old / a_alpha_new)(tau_alpha - t) + t */\
                    auto tau_alpha = reaction_schedule.at(encode(cell_, name));\
                    tau_alpha = (old_a / new_a)*(tau_alpha - age()) + age();\
                    /* 5.d. replace the old tau_alpha in P with the new value */\
                    reaction_schedule.emplace(encode(cell_, name), tau_alpha);\
                } \
            } \
        }
        #include "reactions_list.hpp"
        #undef REACTION

        /*
        #define REACTION(name) {\
            event_id thisReaction = encode(cell_,rid); \
            Minutes current_tau = reaction_schedule[thisReaction];\
            auto r = getRandVariable();\
            auto log_inv_r = -std::log(r);\
            Minutes new_tau = current_tau + Minutes{log_inv_r};\
            reaction_schedule.emplace(thisReaction, new_tau);\
        for (std::size_t i=0; i< propensity_network[rid].size(); i++) { \
            if ( name == propensity_network[rid][i] ) { \
                auto& p = propensities[cell_][name];\
                auto new_p = dense::model::reaction_##name.active_rate(Context(*this, cell_)); \
                total_propensity_ += new_p - p;\
								\
			          \
								event_id cell_reaction_id = encode(cell_, propensity_network[rid][i]); \
								Minutes t_old  = reaction_schedule[cell_reaction_id]; \
								Minutes current_age = age();\
								Minutes t_new = (p/new_p)*(t_old - current_age)+current_age; \
								std::pair<event_id, Minutes> input_pair = std::make_pair(cell_reaction_id,t_new);\
								reaction_schedule.push(input_pair); \
										\
                p = new_p;\
            } \
        } \
    \
        for (std::size_t r=0; r< neighbor_propensity_network[rid].size(); r++) { \
            if (name == neighbor_propensity_network[rid][r]) { \
                for (dense::Natural n=0; n < neighbor_count_by_cell_[cell_]; n++) { \
                    int n_cell = neighbors_by_cell_[cell_][n]; \
                    Context neighbor(*this, n_cell); \
                    auto& p = propensities[n_cell][name];\
                    auto new_p = dense::model::reaction_##name.active_rate(neighbor); \
                    total_propensity_ += new_p - p;\
										\
										auto cell_reaction_id = encode(n_cell,propensity_network[rid][r]);\
										Minutes t_old  = reaction_schedule[cell_reaction_id]; \
										Minutes current_age = age();\
										Minutes t_new = (p/new_p)*(t_old - current_age)+current_age; \
										std::pair<event_id, Minutes> input_pair = std::make_pair(cell_reaction_id,t_new);\
										reaction_schedule.push(input_pair); \
                    p = new_p;\
                } \
            } \
        }\
        }
        #include "reactions_list.hpp"
        #undef REACTION
        /*for (auto rxn : propensity_network[rid]) {
          auto& p = propensities[cell_][rxn];
          auto new_p = dense::model::active_rate(rxn, Context(this, cell_));
          total_propensity_ += new_p - p;
          p = new_p;
        }
        for (auto rxn : neighbor_propensity_network[rid]) {
          for (Natural n = 0; n < neighbor_count_by_cell_[cell_]; ++n) {
            Natural n_cell = neighbors_by_cell_[cell_][n];
            auto& p = propensities[n_cell][rxn];
            auto new_p = dense::model::active_rate(rxn, Context(this, n_cell));
            total_propensity_ += new_p - p;
            p = new_p;
          }
        }*/
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
