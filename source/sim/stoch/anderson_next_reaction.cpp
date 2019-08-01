#include <cmath>
#include "anderson_next_reaction.hpp"
#include "sim/cell_param.hpp"
#include "model_impl.hpp"
#include "core/model.hpp"
#include <limits>
#include <iostream>
#include <cmath>
#include <set>
#include <vector>

namespace dense {
namespace stochastic {

/*
 * SIMULATE
 * main simulation loop
 * notifies observers
 * precondition: t=0
 * postcondition: ti>=time_total
*/

std::uniform_real_distribution<Real> Anderson_Next_Reaction_Simulation::distribution_ = std::uniform_real_distribution<Real>{0.0, 1.0};

CUDA_AGNOSTIC
Minutes Anderson_Next_Reaction_Simulation::age_by (Minutes duration) {
  auto end_time = age() + duration;
  Real& delta_min = delta_minimum;
  Minutes t = age();
  dense::Natural& cell_miu = c_miu;
  auto& reaction_miu = r_miu;
  std::vector<std::vector<Real>> T(cell_count(),std::vector<Real>(NUM_REACTIONS));
  while (t < end_time) {
    t = t + static_cast<Minutes>(delta_min);
    fireReaction(cell_miu,reaction_miu); 
    for (dense::Natural c = 0; c < cell_count(); ++c) {
      for (int i = 0; i < NUM_REACTIONS; ++i) {
        T[c][i] = T[c][i] + propensities[c][i]*delta_minimum;
      }
    }
    Real rand_num = getRandVariable();
    P[cell_miu][reaction_miu] = log(1/rand_num);
    T[cell_miu][reaction_miu] = 0.0;
    delta_min = std::numeric_limits<Real>::infinity();
    for (dense::Natural c = 0; c < cell_count(); ++c) {
      for (int i = 0; i < NUM_REACTIONS; ++i) {
        Real delta_tk= (P[c][i]-T[c][i])/propensities[c][i];
        if (delta_min > delta_tk){
          delta_min = delta_tk;
          cell_miu = c;
          reaction_miu = static_cast<reaction_id>(i);
        }
      }
    }
  }
  return t;
}

Real Anderson_Next_Reaction_Simulation::getRandVariable() {
	return distribution_(generator);
}

/*
 * FIREREACTION
 * fires a reaction by properly decrementing and incrementing its inputs and outputs
 * arg "*c": pointer to a context of the cell to fire the reaction in
 * arg "rid": reaction to fire
*/
void Anderson_Next_Reaction_Simulation::fireReaction(dense::Natural cell, reaction_id rid){
	const reaction_base& r = dense::model::getReaction(rid);
	const specie_id* specie_deltas = r.getSpecieDeltas();
	for (int i=0; i<r.getNumDeltas(); i++){
		update_concentration(cell, specie_deltas[i], r.getDeltas()[i]);
	}
	update_propensities(cell, rid);
}

/*
 * INITPROPENSITIES
 * sets the propensities of each reaction in each cell to its respective active
*/
void Anderson_Next_Reaction_Simulation::initPropensities(){
    for (dense::Natural c = 0; c < cell_count(); ++c) {
        Context ctxt(*this,c);
        #define REACTION(name) \
        propensities[c].push_back(std::max(dense::model::reaction_##name.active_rate(ctxt), Real{0.0001}));
        #include "reactions_list.hpp"
        #undef REACTION
    }
} 

/*
 * INITPROPENSITYNETWORK
 * populates the "propensity_network" and "neighbor_propensity_network" data structures
 * finds inter- and intracellular reactions that have rates affected by the firing of each rxn
*/
void Anderson_Next_Reaction_Simulation::initPropensityNetwork() {

    std::set<specie_id> neighbor_dependencies[NUM_REACTIONS];
    std::set<specie_id> dependencies[NUM_REACTIONS];

    class DependanceContext {
      public:
        DependanceContext(std::set<specie_id>& neighbordeps_tofill,std::set<specie_id>& deps_tofill) :
            interdeps_tofill(neighbordeps_tofill), intradeps_tofill(deps_tofill) {};
        Real getCon(specie_id sp, int = 0) const {
            intradeps_tofill.insert(sp);
            return 0.0;
        };
        Real getCon(specie_id sp){
            intradeps_tofill.insert(sp);
            return 0.0;
        };
        Real getRate(reaction_id) const { return 0.0; };
        Real getDelay(delay_reaction_id) const { return 0.0; };
        Real getCritVal(critspecie_id) const { return 0.0; };
        Real calculateNeighborAvg(specie_id sp, int = 0) const {
            interdeps_tofill.insert(sp);
            return 0.0;
        };
      private:
        std::set<specie_id>& interdeps_tofill;
        std::set<specie_id>& intradeps_tofill;
    };

    #define REACTION(name) \
    const reaction<name>& r##name = dense::model::reaction_##name; \
    r##name.active_rate( DependanceContext (neighbor_dependencies[name],dependencies[name]));
    #include "reactions_list.hpp"
    #undef REACTION

    #define REACTION(name) \
    for (dense::Natural n=0; n<NUM_REACTIONS; n++) { \
        const std::set<specie_id>& intradeps = dependencies[n]; \
        const std::set<specie_id>& interdeps = neighbor_dependencies[n]; \
        std::set<specie_id>::iterator intra = intradeps.begin(); \
        std::set<specie_id>::iterator inter = interdeps.begin(); \
        bool intraRelated = false; \
        bool interRelated = false; \
        for (std::size_t in=0; in<intradeps.size() && !intraRelated; in++){ \
            std::advance(intra, in); \
            for (int o=0; o<r##name.getNumDeltas() && !intraRelated; o++){ \
                 if (r##name.getSpecieDeltas()[o] == *intra) { \
                    intraRelated = true; \
                 } \
            } \
        } \
        for (std::size_t in=0; in<interdeps.size() && !interRelated; in++){ \
            std::advance(inter, in); \
            for (int o=0; o<r##name.getNumDeltas() && !interRelated; o++){ \
                 if (r##name.getSpecieDeltas()[o] == *inter) { \
                    interRelated = true; \
                 } \
            } \
        } \
        if (intraRelated){ \
            propensity_network[name].push_back((reaction_id)n); \
        } \
        if (interRelated){ \
            neighbor_propensity_network[name].push_back((reaction_id)n); \
        } \
    }
    #include "reactions_list.hpp"
    #undef REACTION
}

	
}
}