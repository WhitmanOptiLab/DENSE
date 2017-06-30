#include <cmath>
#include "simulation_stoch.hpp"
#include "cell_param.hpp"
#include "model_impl.hpp"
#include "context_stoch.hpp"
#include "model.hpp"
#include <limits>
#include <iostream>
#include <random>
#include <chrono>
#include <cmath>
#include <cfloat>

typedef std::numeric_limits<double> dbl;
using namespace std;

void simulation_stoch::simulate(){
	RATETYPE analysis_chunks = time_total/analysis_gran;
	
	for (int a=1; a<=analysis_chunks; a++){
		ContextStoch context(*this,0);
		notify(context);

        littleT = 0;
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
	ContextStoch context(*this,0);
	notify(context,true);
}


RATETYPE simulation_stoch::generateTau(){

	ContextStoch c(*this, 0);
	RATETYPE propensity_sum = c.getTotalPropensity();
	
	RATETYPE u = getRandVariable();	
	
	RATETYPE tau = -(log(u))/propensity_sum;

	return tau;

}
/*
void simulation_stoch::generateRXNTaus(RATETYPE tau){

    for (int c=0; c<_cells_total; c++){
        for (int r=0; r<NUM_REACTIONS; r++){

            RATETYPE u = getRandVariable();
            RATETYPE propensity = propensities[c][r];

            RATETYPE rTau = -(log(u))/propensity;

            if (rTau<tau){
                fireOrSchedule(c,(reaction_id)r);
            }
        }
    }
}
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


void simulation_stoch::executeDelayRXN(){
	event delay_rxn = *event_schedule.begin();

   // generateRXNTaus(delay_rxn.time - (t+littleT));

	ContextStoch c(*this, delay_rxn.cell);

	fireReaction(&c,delay_rxn.rxn);

	//t = delay_rxn.time;
    littleT = delay_rxn.time - t;
    //T CHANGED
	event_schedule.erase(delay_rxn);
}


RATETYPE simulation_stoch::getRandVariable(){
	unsigned seed = chrono::system_clock::now().time_since_epoch().count();
	default_random_engine generator (seed);
	uniform_real_distribution<RATETYPE> distribution(0.0,1.0);
	RATETYPE u = distribution(generator);
	return u;
}

void simulation_stoch::tauLeap(RATETYPE tau){
	
   // generateRXNTaus(tau);

	RATETYPE u = getRandVariable();

	ContextStoch context(*this, 0);

	RATETYPE propensity_portion = u * context.getTotalPropensity();

	int j = context.chooseReaction(propensity_portion);
	int r = j%NUM_REACTIONS;
	int c = (j-r)/NUM_REACTIONS;

    fireOrSchedule(c,(reaction_id)r);

    littleT+=tau;
}

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
	
void simulation_stoch::fireReaction(ContextStoch *c, reaction_id rid){
	const reaction_base& r = _model.getReaction(rid);
	const specie_id* inputs = r.getInputs();
    const specie_id* outputs = r.getOutputs();
	for (int i=0; i<r.getNumInputs(); i++){
		c->updateCon(inputs[i],-(r.getInputCounts()[i]));
	}
    for (int o=0;o<r.getNumOutputs(); o++){
        c->updateCon(outputs[o],r.getOutputCounts()[o]);
    }
	c->updatePropensities(rid);
}


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
      for (int r=0; r < NUM_REACTIONS; r++) {
        propensities[c].push_back(0);
      }
    }
    for (int c=0; c<_cells_total;c++){
        ContextStoch ctxt(*this,c);
        for (int r=0; r < NUM_REACTIONS; r++) {
            ctxt.updatePropensities((reaction_id) r);
        }
    }
}

void simulation_stoch::initPropensityNetwork(){
    //Initialize intracellular propensity dependencies
    ContextStoch(*this,0);
    #define REACTION(name) \
    const reaction<name>& r##name = _model.reaction_##name; \
    for (int n=0; n<NUM_REACTIONS; n++) { \
        const reaction_base& r2 = _model.getReaction((reaction_id) n); \
        bool isRelated = false; \
        if (name == n){isRelated = true;} \
        for (int f=0; f<r2.getNumFactors() && !isRelated; f++){ \
            for (int o=0; o<r##name.getNumOutputs() && !isRelated; o++){ \
                if (r##name.getOutputs()[o] == r2.getFactors()[f]){ \
                    isRelated = true; \
                } \
            } \
            for (int i=0; i<r##name.getNumInputs() && !isRelated; i++){ \
                 if (r##name.getInputs()[i] == r2.getFactors()[f]){ \
                    isRelated = true; \
                 } \
            } \
        } \
        for (int in=0; in<r2.getNumInputs() && !isRelated; in++){ \
            for (int o=0; o<r##name.getNumOutputs() && !isRelated; o++){ \
                 if (r##name.getOutputs()[o] == r2.getInputs()[in]){ \
                    isRelated = true; \
                 } \
            } \
            for (int i=0; i<r##name.getNumInputs() && !isRelated; i++){ \
                 if (r##name.getInputs()[i] == r2.getInputs()[in]){ \
                    isRelated = true; \
                 } \
            } \
        } \
        if (isRelated){ \
            propensity_network[name].push_back((reaction_id)n); \
        } \
    }
    #include "reactions_list.hpp"
    #undef REACTION

    //Initialize intercellular propensity dependencies
    vector<specie_id> neighbor_dependencies[NUM_REACTIONS];
    
    class DependanceContext {
      public:
        DependanceContext(vector<specie_id>& neighbordeps_tofill) : 
            deps_tofill(neighbordeps_tofill) {};
        RATETYPE getCon(specie_id sp, int delay=0) const { return 0.0; };
        RATETYPE getRate(reaction_id rid) const { return 0.0; };
        RATETYPE getDelay(delay_reaction_id rid) const { return 0.0; };
        RATETYPE getCritVal(critspecie_id crit) const { return 0.0; };
        RATETYPE calculateNeighborAvg(specie_id sp, int delay=0) const { 
            deps_tofill.push_back(sp);
        };
      private:
        vector<specie_id>& deps_tofill;
    };

    #define REACTION(name) \
    r##name.active_rate(DependanceContext(neighbor_dependencies[name]));

    #include "reactions_list.hpp"
    #undef REACTION


    #define REACTION(name) \
    for (int n=0; n<NUM_REACTIONS; n++) { \
        const vector<specie_id>& deps = neighbor_dependencies[name]; \
        bool isRelated = false; \
        for (int in=0; in<deps.size() && !isRelated; in++){ \
            for (int o=0; o<r##name.getNumOutputs() && !isRelated; o++){ \
                 if (r##name.getOutputs()[o] == deps[in]) { \
                    isRelated = true; \
                 } \
            } \
            for (int i=0; i<r##name.getNumInputs() && !isRelated; i++){ \
                 if (r##name.getInputs()[i] == deps[in]) { \
                    isRelated = true; \
                 } \
            } \
        } \
        if (isRelated){ \
            neighbor_propensity_network[name].push_back((reaction_id)n); \
        } \
    }
    #include "reactions_list.hpp"
    #undef REACTION

}

