#include <cmath>
#include "next_reaction_simulation.hpp"
#include "sim/cell_param.hpp"
#include "model_impl.hpp"
#include "core/model.hpp"
#include <limits>
#include <iostream>
#include <cmath>
#include <set>

namespace dense {
namespace stochastic {

/*
 * SIMULATE
 * main simulation loop
 * notifies observers
 * precondition: t=0
 * postcondition: ti>=time_total
*/

std::uniform_real_distribution<Real> Next_Reaction_Simulation::distribution_ = std::uniform_real_distribution<Real>{0.0, 1.0};

CUDA_AGNOSTIC
Minutes Next_Reaction_Simulation::age_by (Minutes duration) {
  auto end_time = age() + duration;
  while (age() < end_time) {
    //std::cout << "Age: " << age() / Minutes{1} << '\n';
    // 2. Let mu be the reaction whose putative time, tau_mu, stored in P is least.
    auto next_reaction = reaction_schedule.top();
    // 3. Let tau be tau_mu.
    auto tau = next_reaction.second;
    auto cr_pair = decode(next_reaction.first);
    auto c = cr_pair.first;
    auto r = cr_pair.second;
    // 4. Change the number of molecules of reflect execution of reaction mu.
    fireOrSchedule(c, r);
    //    Set t <- tau.
    //std::cout << "Tau: " << tau / Minutes{1} << '\n';
    Simulation::age_by(tau - age());
    // 5. For each edge (mu, alpha) in the dependency graph:
    update_propensities_and_taus(c, r);
    // 6. Go to step 2.
  }
  return age();
}

/*
 * GENERATETAU
 * return "tau": possible timestep leap calculated from a random variable
*/
Minutes Next_Reaction_Simulation::generateTau(Real propensity) {
	auto result = Minutes{ -std::log(getRandVariable()) / propensity };
  if (result.count() < 0) std::cout << "Prop: " << propensity << '\n';
  return result;
}

/*
 * GETSOONESTDELAY
 * return "dTime": the time that the next scheduled delay reaction will fire
 * if no delay reaction is scheduled, the maximum possible float is returned
*/
Minutes Next_Reaction_Simulation::getSoonestDelay() const {
  return reaction_schedule.empty() ?
    Minutes{ std::numeric_limits<Real>::infinity() } :
    reaction_schedule.top().second;
}

Minutes Next_Reaction_Simulation::time_until_next_event() const {
  return getSoonestDelay() - age();
}

/*
 * EXECUTEDELAYRXN
 * calls fireReaction for the next scheduled delay reaction
 * precondition: a delay reaction is scheduled
 * postcondition: the soonest scheduled delay reaction is removed from the schedule
*/
void Next_Reaction_Simulation::executeDelayRXN() {
	std::pair<event_id,Minutes> next_reaction_pair = reaction_schedule.top();
	std::pair<Natural, reaction_id> pair_ids = decode(next_reaction_pair.first);
	
  fireReaction(pair_ids.first, pair_ids.second);
  reaction_schedule.pop(); // TODO: UPDATE, DON"T POP
}

/*
 * GETRANDVARIABLE
 * return "u": a random variable between 0.0 and 1.0
*/

Real Next_Reaction_Simulation::getRandVariable() {
	return distribution_(generator);
}

/*
 * TAULEAP
 * chooses a reaction to fire or schedule and moves forward in time
 * arg "tau": timestep to leap forward by
*/
void Next_Reaction_Simulation::tauLeap(){

	Real propensity_portion = getRandVariable() * get_total_propensity();

	std::pair<Natural, reaction_id> j = choose_reaction();
	reaction_id r =j.second;
	int c = j.first;

    fireOrSchedule(c,r);
}

/*
 * FIREORSCHEDULE
 * fires or schedules a reaction firing in a specific cell
 * arg "c": the cell that the reaction takes place in
 * arg "rid": the reaction to fire or schedule
*/
void Next_Reaction_Simulation::fireOrSchedule(int cell, reaction_id rid){

	delay_reaction_id dri = dense::model::getDelayReactionId(rid);
	
	if (dri!=NUM_DELAY_REACTIONS) {
		event_id rxn_id = encode(cell,rid);
		Minutes reaction_tau = Minutes{ Context(*this, cell).getDelay(dri)};
		std::pair<event_id, Minutes> reaction_pair = std::make_pair(rxn_id,reaction_tau);
		reaction_schedule.push(reaction_pair);
	}
	else {
		fireReaction(cell, rid);
	}
}

/*
 * FIREREACTION
 * fires a reaction by properly decrementing and incrementing its inputs and outputs
 * arg "*c": pointer to a context of the cell to fire the reaction in
 * arg "rid": reaction to fire
*/
void Next_Reaction_Simulation::fireReaction(dense::Natural cell, reaction_id rid){
	const reaction_base& r = dense::model::getReaction(rid);
	const specie_id* specie_deltas = r.getSpecieDeltas();
	for (int i=0; i<r.getNumDeltas(); i++){
		update_concentration(cell, specie_deltas[i], r.getDeltas()[i]);
	}
	update_propensities_and_taus(cell, rid);
}

/*
 * INITPROPENSITIES
 * sets the propensities of each reaction in each cell to its respective active
*/
void Next_Reaction_Simulation::initPropensities(){
    for (dense::Natural c = 0; c < cell_count(); ++c) {
        Context ctxt(*this,c);
        #define REACTION(name) \
        propensities[c].push_back(std::max(dense::model::reaction_##name.active_rate(ctxt), Real{0}));
        #include "reactions_list.hpp"
        #undef REACTION
    }
}

/*
 * INITPROPENSITYNETWORK
 * populates the "propensity_network" and "neighbor_propensity_network" data structures
 * finds inter- and intracellular reactions that have rates affected by the firing of each rxn
*/
void Next_Reaction_Simulation::initPropensityNetwork() {

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


void Next_Reaction_Simulation::initTau() {
  for( dense::Natural c = 0; c < cell_count(); ++c){
    for(auto r = 0; r < NUM_REACTIONS; r++){
      Context ctxt(*this, c);
      auto tau = generateTau(propensities[c][static_cast<reaction_id>(r)]);
      auto crxnid = encode(c, static_cast<reaction_id>(r));
			std::pair<event_id, Minutes> event_pair = std::make_pair(crxnid, tau);
      reaction_schedule.push(event_pair);
    }
  }
}
	
}}


