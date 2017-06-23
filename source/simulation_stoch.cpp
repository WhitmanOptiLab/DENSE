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

typedef std::numeric_limits<double> dbl;
using namespace std;

#define ARRAY_SIZE(array) (sizeof((array))/sizeof((array[0])))

void simulation_stoch::simulate(){
	RATETYPE analysis_chunks = time_total/analysis_gran;
	
	for (int a=1; a<=analysis_chunks; a++){
		ContextStoch context(*this,0);
		notify(context);

		while (t<a*analysis_gran){
		
			RATETYPE tau = generateTau();

			if (t+tau>getSoonestDelay()){
				executeDelayRXN();
			}
			
			else {
				tauLeap(tau);
			}
		}
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

RATETYPE simulation_stoch::getSoonestDelay(){
	event e = *event_schedule.begin();
	return e.time;
}


void simulation_stoch::executeDelayRXN(){
	event delay_rxn = *event_schedule.begin();

	ContextStoch c(*this, delay_rxn.cell);

	fireReaction(&c,delay_rxn.rxn);

	t = delay_rxn.time;
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
	
	RATETYPE u = getRandVariable();

	ContextStoch context(*this, 0);

	RATETYPE propensity_portion = u * context.getTotalPropensity();

	int j = context.chooseReaction(propensity_portion);
	int r = j%NUM_REACTIONS;
	int c = (j-r)/NUM_REACTIONS;

	delay_reaction_id dri = model::getDelayReactionId((reaction_id) r);

	ContextStoch x(*this,c);

	if ((int)dri!=NUM_DELAY_REACTIONS){
		RATETYPE delay = x.getDelay(dri);
		
		event futureRXN;
		futureRXN.time = t + delay;
		futureRXN.rxn = (reaction_id) r;
		futureRXN.cell = c;

		event_schedule.insert(futureRXN);
	}
	else{
		fireReaction(&x,(reaction_id) r);
	}

	t+=tau;
}
	
		
void simulation_stoch::fireReaction(ContextStoch &c, reaction_id rid){
	reaction<rid> r;
	reaction_id outputs[] = *r.getOutputs();
	for (int i=0; i<ARRAY_SIZE(outputs); i++){
		c.updateCon(outputs[i]);
	}
	c.updatePropensities(rid);
}


void simulation_stoch::initialize(){
	
    simulation_base::initialize();

    for (int c = 0; c < _cells_total; c++) {
      vector<int> species;
      concs.push_back(species);
      for (int s = 0; s < NUM_SPECIES; s++) {
        concs[c].push_back(0);
      }
    }
}
