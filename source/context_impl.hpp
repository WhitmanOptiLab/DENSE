// A context defines a locale in which reactions take place and species 
//   reside
#ifndef CONTEXT_IMPL
#define CONTEXT_IMPL

#include "simulation.hpp"
#include "cell_param.hpp"
//#include "context.hpp"
#include <iostream>
#define SQUARE(x) ((x) * (x))
using namespace std;

CPUGPU_FUNC
RATETYPE Context::calculateNeighborAvg(specie_id sp, int delay) const{
    //int NEIGHBORS_2D= _simulation.NEIGHBORS_2D;
    //int neighbors[NUM_DELAY_REACTIONS][NEIGHBORS_2D];
    CPUGPU_TempArray<int, 6>& cells = _simulation._neighbors[sp];

    //memcpy(neighbors[sp.index], _simulation.neighbors[_cell], sizeof(int) * NEIGHBORS_2D);
    //delay = rs[sp][_cell] / _simulation._step_size;
    int time =  _simulation._baby_j[sp] - delay;
    // For each mRNA concentration, average the given cell's neighbors' Delta protein concentrations
    //int* cells = _simulation._neighbors[_cell];
    //int time = WRAP(_simulation._j - delay, _simulation._delay_size[sp.index]);
    // TODO: remove CPDELTA hardcoding
    baby_cl::cell cur_cons = _simulation._baby_cl[pd][time];
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
}

CPUGPU_FUNC
const Context::SpecieRates Context::calculateRatesOfChange(){
    const model& _model = _simulation._model;

    //Step 1: for each reaction, compute reaction rate
    CPUGPU_TempArray<RATETYPE, NUM_REACTIONS> reaction_rates;
    #define REACTION(name) reaction_rates[name] = _model.reaction_##name.active_rate(*this);
        #include "reactions_list.hpp"
    #undef REACTION
    
    //Step 2: allocate specie concentration rate change array
    Context::SpecieRates specie_deltas;
    for (int i = 0; i < NUM_SPECIES; i++) 
      specie_deltas[i] = 0.0;
    
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
void Context::updateCon(const Context::SpecieRates& rates){
    //double step_size= _simulation.step_size;
    
    double curr_rate=0;
    for (int i=0; i< NUM_SPECIES; i++){
        curr_rate= rates[i];
        int baby_j= _simulation._baby_j[i];
        _simulation._baby_cl[i][baby_j+1][_cell]=_simulation._baby_cl[i][baby_j][_cell]+ _simulation._step_size* curr_rate;
    }
    
}

#endif // CONTEXT_IMPL
