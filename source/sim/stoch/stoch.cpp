#include <cmath>
#include "stoch.hpp"
#include "sim/cell_param.hpp"
#include "model_impl.hpp"
#include "stoch_context.hpp"
#include "core/model.hpp"
#include <limits>
#include <iostream>
#include <cmath>
#include <cfloat>
#include <set>

typedef std::numeric_limits<double> dbl;
using namespace std;

/*
 * SIMULATE
 * main simulation loop
 * notifies observers
 * precondition: t=0
 * postcondition: ti>=time_total
*/
void simulation_stoch::simulate(){
	RATETYPE analysis_chunks = time_total/analysis_gran;
	
	for (int a=1; a<=analysis_chunks; a++){
		ContextStoch context(*this,0);
		notify(context);
        littleT = 0;

        if (abort_signaled){
            finalize();
            return;
        }

		while (littleT<analysis_gran && (t+littleT)<time_total){
            RATETYPE tau = generateTau();
			
            if ((t+littleT+tau>getSoonestDelay())&&NUM_DELAY_REACTIONS>0){
                executeDelayRXN();
			}
			else {
				tauLeap(tau);
			}
		}
        t += littleT;
        if (a%int(1.0/analysis_gran)==0)
            cout<<"time="<<t<<endl;
	}
	finalize();
}

/*
 * GENERATETAU
 * return "tau": possible timestep leap calculated from a random variable
*/
RATETYPE simulation_stoch::generateTau(){
	ContextStoch c(*this, 0);
	RATETYPE propensity_sum = c.getTotalPropensity();
	RATETYPE u = getRandVariable();	
	RATETYPE tau = -(log(u))/propensity_sum;

	return tau;
}

/*
 * GETSOONESTDELAY
 * return "dTime": the time that the next scheduled delay reaction will fire
 * if no delay reaction is scheduled, the maximum possible float is returned
*/
RATETYPE simulation_stoch::getSoonestDelay(){
    RATETYPE dTime;
    if (event_schedule.size()>0){
	    event e = *event_schedule.begin();
        dTime = e.time;
    }
    else{
        dTime = FLT_MAX;
    }
	return dTime;
}

/*
 * EXECUTEDELAYRXN
 * calls fireReaction for the next scheduled delay reaction
 * precondition: a delay reaction is scheduled
 * postcondition: the soonest scheduled delay reaction is removed from the schedule
*/
void simulation_stoch::executeDelayRXN(){
	event delay_rxn = *event_schedule.begin();
	
    ContextStoch c(*this, delay_rxn.cell);
	fireReaction(&c,delay_rxn.rxn);
    
    littleT = delay_rxn.time - t;
	
    event_schedule.erase(delay_rxn);
}

/*
 * GETRANDVARIABLE
 * return "u": a random variable between 0.0 and 1.0
*/
RATETYPE simulation_stoch::getRandVariable(){
	uniform_real_distribution<RATETYPE> distribution(0.0,1.0);
	RATETYPE u = distribution(generator);
	return u;
}

/*
 * TAULEAP
 * chooses a reaction to fire or schedule and moves forward in time
 * arg "tau": timestep to leap forward by
*/
void simulation_stoch::tauLeap(RATETYPE tau){
	
	RATETYPE u = getRandVariable();

	ContextStoch context(*this, 0);

	RATETYPE propensity_portion = u * context.getTotalPropensity();

	int j = context.chooseReaction(propensity_portion);
	int r = j%NUM_REACTIONS;
	int c = (j-r)/NUM_REACTIONS;

    fireOrSchedule(c,(reaction_id)r);

    littleT+=tau;
}

/*
 * FIREORSCHEDULE
 * fires or schedules a reaction firing in a specific cell
 * arg "c": the cell that the reaction takes place in
 * arg "rid": the reaction to fire or schedule
*/
void simulation_stoch::fireOrSchedule(int c, reaction_id rid){

	delay_reaction_id dri = model::getDelayReactionId(rid);

	ContextStoch x(*this,c);

	if (dri!=NUM_DELAY_REACTIONS){
		RATETYPE delay = x.getDelay(dri);
		
		event futureRXN;
		futureRXN.time = t + littleT + delay;
		futureRXN.rxn = rid;
		futureRXN.cell = c;

		event_schedule.insert(futureRXN);
	}
	else{
		fireReaction(&x,rid);
	}
}

/*
 * FIREREACTION
 * fires a reaction by properly decrementing and incrementing its inputs and outputs
 * arg "*c": pointer to a context of the cell to fire the reaction in
 * arg "rid": reaction to fire
*/
void simulation_stoch::fireReaction(ContextStoch *c, reaction_id rid){
	const reaction_base& r = _model.getReaction(rid);
	const specie_id* specie_deltas = r.getSpecieDeltas();
	for (int i=0; i<r.getNumDeltas(); i++){
		c->updateCon(r.getSpecieDeltas()[i], r.getDeltas()[i]);
	}
	c->updatePropensities(rid);
}

/*
 * INITIALIZE
 * calls "simulation_base" initialize function
 * populates main data structures "concs", "propensities"
 * precondition: propensities and concs are empty vectors
*/
void simulation_stoch::initialize(){
	
    simulation_base::initialize();

    initPropensityNetwork();

    for (int c = 0; c < _cells_total; c++) {
      vector<int> species;
      vector<RATETYPE> props;
      concs.push_back(species);
      propensities.push_back(props);
      for (int s = 0; s < NUM_SPECIES; s++) {
        concs[c].push_back(0);
      }
    }
    initPropensities(); 
}

/*
 * INITPROPENSITIES
 * sets the propensities of each reaction in each cell to its respective active
*/
void simulation_stoch::initPropensities(){
    for (int c=0; c<_cells_total; c++){
        ContextStoch ctxt(*this,c);
        #define REACTION(name) \
        propensities[c].push_back(_model.reaction_##name.active_rate(ctxt));
        #include "reactions_list.hpp"
        #undef REACTION
    }
}

/*
 * INITPROPENSITYNETWORK
 * populates the "propensity_network" and "neighbor_propensity_network" data structures
 * finds inter- and intracellular reactions that have rates affected by the firing of each rxn
*/
void simulation_stoch::initPropensityNetwork(){
   
    set<specie_id> neighbor_dependencies[NUM_REACTIONS];
    set<specie_id> dependencies[NUM_REACTIONS];
    
    class DependanceContext {
      public:
        DependanceContext(set<specie_id>& neighbordeps_tofill,set<specie_id>& deps_tofill) : 
            interdeps_tofill(neighbordeps_tofill), intradeps_tofill(deps_tofill) {};
        RATETYPE getCon(specie_id sp, int delay=0) const {
            intradeps_tofill.insert(sp);
        };
        RATETYPE getCon(specie_id sp){
            intradeps_tofill.insert(sp);
        };
        RATETYPE getRate(reaction_id rid) const { return 0.0; };
        RATETYPE getDelay(delay_reaction_id rid) const { return 0.0; };
        RATETYPE getCritVal(critspecie_id crit) const { return 0.0; };
        RATETYPE calculateNeighborAvg(specie_id sp, int delay=0) const { 
            interdeps_tofill.insert(sp);
        };
      private:
        set<specie_id>& interdeps_tofill;
        set<specie_id>& intradeps_tofill;
    };

    #define REACTION(name) \
    const reaction<name>& r##name = _model.reaction_##name; \
    r##name.active_rate( DependanceContext (neighbor_dependencies[name],dependencies[name]));
    #include "reactions_list.hpp"
    #undef REACTION

    #define REACTION(name) \
    for (int n=0; n<NUM_REACTIONS; n++) { \
        const set<specie_id>& intradeps = dependencies[n]; \
        const set<specie_id>& interdeps = neighbor_dependencies[n]; \
        std::set<specie_id>::iterator intra = intradeps.begin(); \
        std::set<specie_id>::iterator inter = interdeps.begin(); \
        bool intraRelated = false; \
        bool interRelated = false; \
        for (int in=0; in<intradeps.size() && !intraRelated; in++){ \
            std::advance(intra, in); \
            for (int o=0; o<r##name.getNumDeltas() && !intraRelated; o++){ \
                 if (r##name.getSpecieDeltas()[o] == *intra) { \
                    intraRelated = true; \
                 } \
            } \
        } \
        for (int in=0; in<interdeps.size() && !interRelated; in++){ \
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

