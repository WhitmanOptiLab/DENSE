// A context defines a locale in which reactions take place and species 
//   reside

#include "simulation.hpp"
#include "cell_param.hpp"
#include "context.hpp"
#include <iostream>
#define SQUARE(x) ((x) * (x))
using namespace std;

#if 0
RATETYPE Context::calculateNeighbourAvg(specie_id sp) const{
    //int NEIGHBORS_2D= _simulation.NEIGHBORS_2D;
    //int neighbors[NUM_DELAY_REACTIONS][NEIGHBORS_2D];
    std::array<int, 6> cells = _simulation._neighbors[sp];

    //memcpy(neighbors[sp.index], _simulation.neighbors[_cell], sizeof(int) * NEIGHBORS_2D);
    //delay = rs[sp][_cell] / _simulation._step_size;
    int time =  _simulation._baby_j[sp]- (_simulation._delays[sp][_cell]/ _simulation._step_size);
    // For each mRNA concentration, average the given cell's neighbors' Delta protein concentrations
    //int* cells = _simulation._neighbors[_cell];
    //int time = WRAP(_simulation._j - delay, _simulation._delay_size[sp.index]);
    // TODO: remove CPDELTA hardcoding
    baby_cl::cell cur_cons = _simulation._baby_cl[pd][time];
    RATETYPE sum=0;
    //since the tissue is not growing now
    //start is 0 and end is 10, instead of_simulation.active_start_record[time] and_simulation.active_end_record[time]
    if (_cell % _simulation.width_total == 0) {
        sum = (cur_cons[cells[0]] + cur_cons[cells[3]] + cur_cons[cells[4]] + cur_cons[cells[5]]) / 4;
    } else if (_cell % _simulation.width_total == 10) {
        sum = (cur_cons[cells[0]] + cur_cons[cells[1]] + cur_cons[cells[2]] + cur_cons[cells[3]]) / 4;
    } else {
        sum = (cur_cons[cells[0]] + cur_cons[cells[1]] + cur_cons[cells[2]] + cur_cons[cells[3]] + cur_cons[cells[4]] + cur_cons[cells[5]]) / 6;
    }
    return sum;
}
#endif

const std::array<RATETYPE, NUM_SPECIES> Context::calculateRatesOfChange(){
    const model& _model = _simulation._model;

    //Step 1: for each reaction, compute reaction rate
    std::array<RATETYPE, NUM_REACTIONS> reaction_rates;
    #define REACTION(name) reaction_rates[name] = _model.reaction_##name.active_rate(*this);
        #include "reactions_list.hpp"
    #undef REACTION
    
    //Step 2: allocate specie concentration rate change array
    std::array<RATETYPE, NUM_SPECIES> specie_deltas;
    for (int i = 0; i < NUM_SPECIES; i++) 
      specie_deltas[i] = 0.0;
    
    //Step 3: for each reaction rate, for each specie it affects, accumulate its contributions
    
    #define REACTION(name) \
    const reaction<name>& r##name = _model.reaction_##name; \
    for (int j = 0; j < r##name.getNumInputs(); j++) { \
        specie_deltas[r##name.getInputs()[j]] -= reaction_rates[name]*r##name.getInputCounts()[j]; \
    } \
    for (int j = 0; j < _model.reaction_##name.getNumOutputs(); j++) { \
        specie_deltas[r##name.getOutputs()[j]] += reaction_rates[name]*r##name.getOutputCounts()[j]; \
    }
    #include "reactions_list.hpp"
    #undef REACTION
    
    return specie_deltas;
}

void Context::updateCon(const std::array<RATETYPE, NUM_SPECIES>& rates){
    //double step_size= _simulation.step_size;
    
    double curr_rate=0;
    for (int i=0; i< NUM_SPECIES; i++){
        curr_rate= rates[i];
        int baby_j= _simulation._baby_j[i];
        _simulation._cl[i][baby_j+1][_cell]=_simulation._cl[i][baby_j][_cell]+ _simulation._step_size* curr_rate;
    }
    
}


RATETYPE Context::cal_avgpd(specie_id sp) const{
    std::array<int, 6> cells = _simulation._neighbors[sp];
    
    //memcpy(neighbors[sp.index], _simulation.neighbors[_cell], sizeof(int) * NEIGHBORS_2D);
    //delay = rs[sp][_cell] / _simulation._step_size;
    int time =  _simulation._baby_j[sp]- (_simulation._delays[sp][_cell]/ _simulation._step_size);
    // For each mRNA concentration, average the given cell's neighbors' Delta protein concentrations
    //int* cells = _simulation._neighbors[_cell];
    //int time = WRAP(_simulation._j - delay, _simulation._delay_size[sp.index]);
    // TODO: remove CPDELTA hardcoding
    baby_cl::cell cur_cons = _simulation._baby_cl[pd][time];
    RATETYPE avg_delay=0;
    //since the tissue is not growing now
    //start is 0 and end is 10, instead of_simulation.active_start_record[time] and_simulation.active_end_record[time]
    if (_cell % _simulation._width_total == 0) {
        avg_delay = (cur_cons[cells[0]] + cur_cons[cells[3]] + cur_cons[cells[4]] + cur_cons[cells[5]]) / 4;
    } else if (_cell % _simulation._width_total == 10) {
        avg_delay = (cur_cons[cells[0]] + cur_cons[cells[1]] + cur_cons[cells[2]] + cur_cons[cells[3]]) / 4;
    } else {
        avg_delay = (cur_cons[cells[0]] + cur_cons[cells[1]] + cur_cons[cells[2]] + cur_cons[cells[3]] + cur_cons[cells[4]] + cur_cons[cells[5]]) / 6;
    }
    return avg_delay;
}

RATETYPE Context::cal_transcription(specie_id sp) const{
    RATETYPE th1h1, tm1m1, tm1m2, tm2m2, tdelta;
    RATETYPE numerator, denominator;
    if (sp == md){
        tdelta = 0;
    } else {
        tdelta = cal_tdelta(sp);
    }
    switch(sp){
        case mm2:
            tm1m1 = cal_tdimer(pm11);
            tm1m2 = cal_tdimer(pm12);
            tm2m2 = cal_tdimer(pm22);
            denominator = 1 + tdelta + SQUARE(tm1m1) + SQUARE(tm1m2) + SQUARE(tm2m2);
            break;
        default:
            th1h1 = cal_tdimer(ph11);
            denominator = 1 + tdelta + SQUARE(th1h1);
            break;
    }
    switch (sp) {
        case mm1:
            numerator = tdelta;
            break;
        default:
            numerator = 1 + tdelta;
            break;
    }
    
    return numerator/denominator;
}

RATETYPE Context::cal_tdelta(specie_id sp) const{
    return _simulation._critValues[1][_cell] == 0 ? 0 : cal_avgpd(sp) / _simulation._critValues[1][_cell];
}

RATETYPE Context::cal_tdimer(specie_id sp) const{
    int critVaIdx;
    switch(sp){
        case ph11:
            critVaIdx = 0;
            break;
        case pm11:
            critVaIdx = 2;
            break;
        case pm12:
            critVaIdx = 3;
            break;
        case pm22:
            critVaIdx = 4;
            break;        
        default:
            break;
    }
    return _simulation._critValues[critVaIdx][_cell] == 0 ? 0 : getCon(sp) / _simulation._critValues[critVaIdx][_cell];
}




