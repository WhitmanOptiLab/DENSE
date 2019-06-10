#include <cmath>
#include "fast_gillespie_direct_simulation.hpp"
#include "sim/cell_param.hpp"
#include "model_impl.hpp"
#include "core/model.hpp"
#include <limits>
#include <iostream>
#include <cmath>
#include <set>

namespace dense {

/*
 * SIMULATE
 * main simulation loop
 * notifies observers
 * precondition: t=0
 * postcondition: ti>=time_total
*/

std::uniform_real_distribution<Real> Fast_Gillespie_Direct_Simulation::distribution_ = std::uniform_real_distribution<Real>{0.0, 1.0};

CUDA_AGNOSTIC
Minutes Fast_Gillespie_Direct_Simulation::age_by (Minutes duration) {
  auto end_time = age() + duration;
  while (age() < end_time) {
    Minutes tau, t_until_event;

    while ((tau = generateTau()) > (t_until_event = time_until_next_event())) {
      Simulation::age_by(t_until_event);
      executeDelayRXN();
      if (age() >= end_time) return age();
    }

    tauLeap();
    Simulation::age_by(tau);
  }
  return age();
}

/*
 * GENERATETAU
 * return "tau": possible timestep leap calculated from a random variable
*/
Minutes Fast_Gillespie_Direct_Simulation::generateTau() {
  auto r = getRandVariable();
  auto log_inv_r = -std::log(r);

	return Minutes{ log_inv_r / get_total_propensity() };
}

/*
 * GETSOONESTDELAY
 * return "dTime": the time that the next scheduled delay reaction will fire
 * if no delay reaction is scheduled, the maximum possible float is returned
*/
Minutes Fast_Gillespie_Direct_Simulation::getSoonestDelay() const {
  return event_schedule.empty() ?
    Minutes{ std::numeric_limits<Real>::infinity() } :
    event_schedule.top().time;
}

Minutes Fast_Gillespie_Direct_Simulation::time_until_next_event() const {
  return getSoonestDelay() - age();
}

/*
 * EXECUTEDELAYRXN
 * calls fireReaction for the next scheduled delay reaction
 * precondition: a delay reaction is scheduled
 * postcondition: the soonest scheduled delay reaction is removed from the schedule
*/
void Fast_Gillespie_Direct_Simulation::executeDelayRXN() {
  event delay_rxn = event_schedule.top();
  fireReaction(delay_rxn.cell, delay_rxn.rxn);
  event_schedule.pop();
}

/*
 * GETRANDVARIABLE
 * return "u": a random variable between 0.0 and 1.0
*/

Real Fast_Gillespie_Direct_Simulation::getRandVariable() {
	return distribution_(generator);
}

/*
 * TAULEAP
 * chooses a reaction to fire or schedule and moves forward in time
 * arg "tau": timestep to leap forward by
*/
void Fast_Gillespie_Direct_Simulation::tauLeap(){

	Real propensity_portion = getRandVariable() * get_total_propensity();

	int j = choose_reaction(propensity_portion);
	int r = j % NUM_REACTIONS;
	int c = j / NUM_REACTIONS;

    fireOrSchedule(c,(reaction_id)r);
}

/*
 * FIREORSCHEDULE
 * fires or schedules a reaction firing in a specific cell
 * arg "c": the cell that the reaction takes place in
 * arg "rid": the reaction to fire or schedule
*/
void Fast_Gillespie_Direct_Simulation::fireOrSchedule(int cell, reaction_id rid){

	delay_reaction_id dri = dense::model::getDelayReactionId(rid);

	if (dri!=NUM_DELAY_REACTIONS) {
		event_schedule.push({ age() + Minutes{ Context(*this, cell).getDelay(dri) }, cell, rid });
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
void Fast_Gillespie_Direct_Simulation::fireReaction(dense::Natural cell, reaction_id rid){
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
void Fast_Gillespie_Direct_Simulation::initPropensities(){
   total_propensity_ = 0.0;
    for (dense::Natural c = 0; c < cell_count(); ++c) {
        Context ctxt(*this,c);
        #define REACTION(name) \
        propensities[c].push_back(dense::model::reaction_##name.active_rate(ctxt));\
        total_propensity_ += propensities[c].back();
        #include "reactions_list.hpp"
        #undef REACTION
    }
}

/*
 * INITPROPENSITYNETWORK
 * populates the "propensity_network" and "neighbor_propensity_network" data structures
 * finds inter- and intracellular reactions that have rates affected by the firing of each rxn
*/
void Fast_Gillespie_Direct_Simulation::initPropensityNetwork() {

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
