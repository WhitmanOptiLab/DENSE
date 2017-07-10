// A context defines a locale in which reactions take place and species 
//   reside
#ifndef CONTEXT_IMPL
#define CONTEXT_IMPL

#include "common_utils.hpp"
#include "simulation_determ.hpp"
#include "cell_param.hpp"
//#include "context.hpp"
#include <iostream>
#define SQUARE(x) ((x) * (x))
using namespace std;

//declare reaction inits here
#define REACTION(name) \
  template<> \
  reaction< name >::reaction() : \
    reaction_base( num_inputs_##name, num_outputs_##name, \
    num_factors_##name, in_counts_##name, out_counts_##name, \
    inputs_##name, outputs_##name, factors_##name){}
#include "reactions_list.hpp"
#undef REACTION

CPUGPU_FUNC
RATETYPE simulation_determ::Context::calculateNeighborAvg(specie_id sp, int delay) const{
    //int NEIGHBORS_2D= _simulation.NEIGHBORS_2D;
    //memcpy(neighbors[sp.index], _simulation.neighbors[_cell], sizeof(int) * NEIGHBORS_2D);
    //delay = rs[sp][_cell] / _simulation._step_size;
    // For each mRNA concentration, average the given cell's neighbors' Delta protein concentrations
    //int* cells = _simulation._neighbors[_cell];
    //int time = WRAP(_simulation._j - delay, _simulation._delay_size[sp.index]);
    // TODO: remove CPDELTA hardcoding

    RATETYPE sum=0;
    
    for (int i=0; i<_simulation._numNeighbors[_cell]; i++){
        sum+=_simulation._baby_cl[sp][-delay][_simulation._neighbors[_cell][i]];
    }
    RATETYPE avg = sum/_simulation._numNeighbors[_cell]; 
    return avg;
}

CPUGPU_FUNC
const simulation_determ::Context::SpecieRates simulation_determ::Context::calculateRatesOfChange(){
    const model& _model = _simulation._model;
    
    //Step 1: for each reaction, compute reaction rate
    CPUGPU_TempArray<RATETYPE, NUM_REACTIONS> reaction_rates;
    #define REACTION(name) reaction_rates[name] = _model.reaction_##name.active_rate(*this);
        #include "reactions_list.hpp"
    #undef REACTION
    
    //Step 2: allocate specie concentration rate change array
    SpecieRates specie_deltas;
    for (int i = 0; i < NUM_SPECIES; i++) 
      specie_deltas[i] = 0;
    
    //Step 3: for each reaction rate, for each specie it affects, accumulate its contributions
    #define REACTION(name) \
    const reaction<name>& r##name = _model.reaction_##name; \
    for (int j = 0; j < r##name.getNumInputs(); j++) { \
        specie_deltas[inputs_##name[j]] -= reaction_rates[name]*in_counts_##name[j]; \
    } \
    for (int j = 0; j < _model.reaction_##name.getNumOutputs(); j++) { \
        specie_deltas[outputs_##name[j]] += reaction_rates[name]*out_counts_##name[j]; \
    }
    #include "reactions_list.hpp"
    #undef REACTION
    
    return specie_deltas;
}

CPUGPU_FUNC
void simulation_determ::Context::updateCon(const simulation_determ::Context::SpecieRates& rates){
    //double step_size= _simulation.step_size;
    
    double curr_rate=0;
    for (int i=0; i< NUM_SPECIES; i++){
        curr_rate= rates[i];
        _simulation._baby_cl[i][1][_cell]=_simulation._baby_cl[i][0][_cell]+ _simulation._step_size* curr_rate;
    }
}

#endif // CONTEXT_IMPL
