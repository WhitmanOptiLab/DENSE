// A context defines a locale in which reactions take place and species 
//   reside
#ifndef CONTEXT_STOCH
#define CONTEXT_STOCH

#include "simulation_stoch.hpp"
#include "cell_param.hpp"
#include "reaction.hpp"
//#include "context.hpp"
#include <iostream>
using namespace std;

CPUGPU_FUNC
RATETYPE simulation_stoch::ContextStoch::calculateNeighborAvg(specie_id sp) const{
    /*
    //int NEIGHBORS_2D= _simulation.NEIGHBORS_2D;
    //int neighbors[NUM_DELAY_REACTIONS][NEIGHBORS_2D];
    CPUGPU_TempArray<int, 6>& cells = _simulation._neighbors[_cell];

    //memcpy(neighbors[sp.index], _simulation.neighbors[_cell], sizeof(int) * NEIGHBORS_2D);
    //delay = rs[sp][_cell] / _simulation._step_size;
    // For each mRNA concentration, average the given cell's neighbors' Delta protein concentrations
    //int* cells = _simulation._neighbors[_cell];
    //int time = WRAP(_simulation._j - delay, _simulation._delay_size[sp.index]);
    // TODO: remove CPDELTA hardcoding
    baby_cl::cell cur_cons = _simulation.concs[pd][-delay];
    RATETYPE sum=0;
    //since the tissue is not growing now
    //start is 0 and end is 10, instead of_simulation.active_start_record[time] and_simulation.active_end_record[time]
    if (_cell % _simulation._width_total == 0) {
        sum = (cur_cons[cells[0]] + cur_cons[cells[3]] + cur_cons[cells[4]] + cur_cons[cells[5]]) / 4;
    } else if (_cell % _simulation._width_total == 10) {
        sum = (cur_cons[cells[0]] + cur_cons[cells[1]] + cur_cons[cells[2]] + cur_cons[cells[3]]) / 4;
    } else {
        sum = (cur_cons[cells[0]] + cur_cons[cells[1]] + cur_cons[cells[2]] + cur_cons[cells[3]] + cur_cons[cells[4]] + cur_cons[cells[5]]) / 6;
    }
    return sum;
    */
    RATETYPE sum=0;

    for (int i; i<_simulation._neighbors[_cell].size(); i++){
        sum+=_simulation.concs[_simulation._neighbors[_cell][i]][sp];
    }
    
    return sum;
}

CPUGPU_FUNC
void simulation_stoch::ContextStoch::updatePropensities(reaction_id rid){
    const model& _model = _simulation._model;

    //reaction<rxn> rxn;

    //Step 1: for each reaction, compute reaction rate
    CPUGPU_TempArray<RATETYPE, NUM_REACTIONS> reaction_rates;
    #define REACTION(name) reaction_rates[name] = _model.reaction_##name.active_rate(*this);
        #include "reactions_list.hpp"
    #undef REACTION
    
    //Step 2: allocate specie concentration rate change array
    for (int i = 0; i < NUM_SPECIES; i++) 
      propensities[_cell][i] = 0;
    
    //Step 3: for each reaction rate, for each specie it affects, accumulate its contributions
    #define REACTION(name) \
    const reaction<name>& r##name = _model.reaction_##name; \
    for (int j = 0; j < r##name.getNumInputs(); j++) { \
        propensities[_cell][inputs_##name[j]] -= reaction_rates[name]*in_counts_##name[j]; \
    } \
    for (int j = 0; j < _model.reaction_##name.getNumOutputs(); j++) { \
        propensities[_cell][outputs_##name[j]] += reaction_rates[name]*out_counts_##name[j]; \
    }
    #include "reactions_list.hpp"
    #undef REACTION
}

CPUGPU_FUNC
RATETYPE simulation_stoch::ContextStoch::getTotalPropensity(){
    RATETYPE sum = 0;
    for (int c=0; c<_simulation._cells_total; c++){
      for (int s=0; s<NUM_SPECIES; s++){
        sum+=propensities[c][s];
      }
    }
    return sum;
}

CPUGPU_FUNC
int simulation_stoch::ContextStoch::chooseReaction(RATETYPE propensity_portion){
    RATETYPE sum;
    int c,s;

    for (c=0; c<_simulation._cells_total; c++){
      for (s=0; s<NUM_SPECIES; s++){
        sum+=propensities[c][s];
	if (sum<propensity_portion){
	  return (c*NUM_SPECIES)+s;
	}
      }
    }
    return (c*NUM_SPECIES)+s;
}


#endif // CONTEXT_STOCH
