// A context defines a locale in which reactions take place and species 
//   reside

#include "simulation.hpp"
#include "rates.hpp"
#include "contexts.hpp"
#include "model_impl.hpp"
#include <iostream>


using namespace std;

double context::calculateNeighbourAvg(specie sp, int time){
    int NEIGHBORS_2D= _simulation.NEIGHBORS_2D;
    int neighbors[NUM_DELAY_REACTIONS][NEIGHBORS_2D];
 
    //memcpy(neighbors[sp.index], _simulation.neighbors[_cell], sizeof(int) * NEIGHBORS_2D);
    delay = rs[sp.index][_cell] / _simulation.step_size;
    
    // For each mRNA concentration, average the given cell's neighbors' Delta protein concentrations
    int* cells = _simulation._neighbors[_cell];
    //int time = WRAP(_simulation._j - delay, _simulation._delay_size[sp.index]);
    concentration_level<double>::cell cur_cons = _simulation._cl[CPDELTA][time];
    double sum=0;
    if (_cell % _simulation.width_total == _simulation.active_start_record[time]) {
        sum = (cur_cons[cells[0]] + cur_cons[cells[3]] + cur_cons[cells[4]] + cur_cons[cells[5]]) / 4;
    } else if (_cell % _simulation.width_total == _simulation.active_start_record[time]) {
        sum = (cur_cons[cells[0]] + cur_cons[cells[1]] + cur_cons[cells[2]] + cur_cons[cells[3]]) / 4;
    } else {
        sum = (cur_cons[cells[0]] + cur_cons[cells[1]] + cur_cons[cells[2]] + cur_cons[cells[3]] + cur_cons[cells[4]] + cur_cons[cells[5]]) / 6;
    }

    return sum;
}

double[] calculateRates(){
    //Step 1: for each reaction, compute reaction rate
    E reaction_rates[NUM_REACTIONS];
    #define REACTION(name) reaction_rates[name] = _model.reaction_##name.active_rate(c);
        #include "reaction_list.hpp"
    #undef REACTION
    
    //Step 2: allocate specie concentration rate change array
    std::array<E, NUM_SPECIES> specie_deltas;
    for (int i = 0; i < NUM_SPECIES; i++) specie_deltas[i] = 0.0;
    
    //Step 3: for each reaction rate, for each specie it affects, accumulate its contributions
    
    #define REACTION(name) \
    for (int j = 0; j < _model.reaction_##name.num_inputs; j++) { \
        specie_deltas[name.inputs[j]] -= reaction_rates[name]*_model.reaction_##name.in_counts[j]; \
    } \
    for (int j = 0; j < _model.reaction_##name.num_outputs; j++) { \
        specie_deltas[name.outputs[j]] += reaction_rates[name]*_model.reaction_##name.out_counts[j]; \
    }
    #include "reaction_list.hpp"
    #undef REACTION(name)
    
    return specie_deltas;
}

void context::updateCon(double[] rates){
    //double step_size= _simulation.step_size;
    
    double curr_rate=0;
    for (int i=0; i< NUM_SPECIES; i++){
        curr_rate= rates[i];
        int baby_j= _simulation._baby_j[i];
        _simulation._cl[i][baby_j+1][_cell]=_simulation._cl[i][baby_j][cell]+ _simulation.step_size* curr_rate;
    }
    
}
