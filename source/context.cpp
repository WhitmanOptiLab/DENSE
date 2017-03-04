// A context defines a locale in which reactions take place and species 
//   reside

#include "simulation.hpp"
#include "cell_param.hpp"
#include "context.hpp"
#include <iostream>

using namespace std;

#if 0
RATETYPE Context::calculateNeighbourAvg(specie_id sp, int delay){
    int NEIGHBORS_2D= _simulation.NEIGHBORS_2D;
    int neighbors[NUM_DELAY_REACTIONS][NEIGHBORS_2D];
 
    //memcpy(neighbors[sp.index], _simulation.neighbors[_cell], sizeof(int) * NEIGHBORS_2D);
    delay = rs[sp.index][_cell] / _simulation.step_size;
    
    // For each mRNA concentration, average the given cell's neighbors' Delta protein concentrations
    int* cells = _simulation._neighbors[_cell];
    //int time = WRAP(_simulation._j - delay, _simulation._delay_size[sp.index]);
    // TODO: remove CPDELTA hardcoding
    concentration_level<RATETYPE>::cell cur_cons = _simulation.baby_cl[CPDELTA][time];
    RATETYPE sum=0;
    if (_cell % _simulation.width_total == _simulation.active_start_record[time]) {
        sum = (cur_cons[cells[0]] + cur_cons[cells[3]] + cur_cons[cells[4]] + cur_cons[cells[5]]) / 4;
    } else if (_cell % _simulation.width_total == _simulation.active_start_record[time]) {
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
        _simulation._cl[i][baby_j+1][_cell]=_simulation._cl[i][baby_j][_cell]+ _simulation.step_size* curr_rate;
    }
    
}


RATETYPE Context::cal_avgpd(int mRNA_idx) const{
    //3 = number of mrna that is delta dependent
    
        std::array<int, 6> cells = _simulation._neighbors[mRNA_idx];
        //int cell = old_cells_mrna[IMH1 + j];
        int time =  _simulation._baby_j[mRNA_idx]- _simulation._delays[mRNA_idx][_cell];
        baby_cl::cell cur_cons = _simulation._baby_cl[pd][time];
        double avg_delay = 0;
        if (_cell % _simulation._width_total == cl.active_start_record[time]) {
            avg_delay = (cur_cons[cells[0]] + cur_cons[cells[3]] + cur_cons[cells[4]] + cur_cons[cells[5]]) / 4;
        } else if (_cell % _simulation._width_total == cl.active_end_record[time]) {
            avg_delay = (cur_cons[cells[0]] + cur_cons[cells[1]] + cur_cons[cells[2]] + cur_cons[cells[3]]) / 4;
        } else {
            avg_delay = (cur_cons[cells[0]] + cur_cons[cells[1]] + cur_cons[cells[2]] + cur_cons[cells[3]] + cur_cons[cells[4]] + cur_cons[cells[5]]) / 6;
        }
        //avg_delays = sum;
    
    
    return avg_delay;
}

RATETYPE Context::cal_transcription_her1() const{
    RATETYPE th1h1= 0, tdelta;
    th1h1 = _simulation._critValues[0][_cell] == 0 ? 0 : getCon(ph11) / _simulation._critValues[0][_cell];
    tdelta = _simulation._critValues[1][_cell] == 0 ? 0 : cal_avgpd(mh1) / _simulation._critValues[1][_cell];
    return ((1 + tdelta) / (1 + tdelta + th1h1*th1h1));
}






