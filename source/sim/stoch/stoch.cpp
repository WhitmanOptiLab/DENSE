#include <cmath>
#include "stoch.hpp"
#include "sim/cell_param.hpp"
#include "model_impl.hpp"
#include "core/model.hpp"
#include <limits>
#include <iostream>
#include <cmath>
#include <cfloat>
#include <set>

namespace dense {

/*
 * SIMULATE
 * main simulation loop
 * notifies observers
 * precondition: t=0
 * postcondition: ti>=time_total
*/

void Stochastic_Simulation::simulate_for (Real duration) {
  Real end_time = age_ + duration;
  while (age_ < end_time) {
    Real tau;

    while (age_ + (tau = generateTau()) > getSoonestDelay()) {
      age_ = getSoonestDelay();
      executeDelayRXN();
      if (age_ >= end_time) return;
    }

    tauLeap();
    age_ += tau;
  }
}

/*
 * GENERATETAU
 * return "tau": possible timestep leap calculated from a random variable
*/
Real Stochastic_Simulation::generateTau() {
	return -std::log(getRandVariable()) / get_total_propensity();
}

/*
 * GETSOONESTDELAY
 * return "dTime": the time that the next scheduled delay reaction will fire
 * if no delay reaction is scheduled, the maximum possible float is returned
*/
Real Stochastic_Simulation::getSoonestDelay() {
  return event_schedule.empty() ?
    std::numeric_limits<Real>::max() :
    event_schedule.top().time;
}

/*
 * EXECUTEDELAYRXN
 * calls fireReaction for the next scheduled delay reaction
 * precondition: a delay reaction is scheduled
 * postcondition: the soonest scheduled delay reaction is removed from the schedule
*/
void Stochastic_Simulation::executeDelayRXN(){
	event delay_rxn = event_schedule.top();
	fireReaction(delay_rxn.cell, delay_rxn.rxn);
  event_schedule.pop();
}

/*
 * GETRANDVARIABLE
 * return "u": a random variable between 0.0 and 1.0
*/

Real Stochastic_Simulation::getRandVariable() {
	static std::uniform_real_distribution<Real> distribution(0.0, 1.0);
	return distribution(generator);
}

/*
 * TAULEAP
 * chooses a reaction to fire or schedule and moves forward in time
 * arg "tau": timestep to leap forward by
*/
void Stochastic_Simulation::tauLeap(){

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
void Stochastic_Simulation::fireOrSchedule(int cell, reaction_id rid){

	delay_reaction_id dri = dense::model::getDelayReactionId(rid);

	if (dri!=NUM_DELAY_REACTIONS) {
		event_schedule.push({ age_ + Context(*this, cell).getDelay(dri), cell, rid });
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
void Stochastic_Simulation::fireReaction(dense::Natural cell, reaction_id rid){
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
void Stochastic_Simulation::initPropensities(){
   total_propensity_ = 0.0;
    for (dense::Natural c = 0; c < _cells_total; ++c) {
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
void Stochastic_Simulation::initPropensityNetwork() {

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
