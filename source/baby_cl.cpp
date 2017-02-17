#include "baby_cl.hpp"

using namespace std;
RATETYPE calc_delay(int relatedReactions[]){
    RATETYPE max;
    for (int j = 0; j <= sizeof(&relatedReactions); j++) {
        for (int k = 0; k < _sim.width_total; k++) {
            // Calculate the minimum delay, accounting for the maximum allowable perturbation and gradients
            max = MAX(max, (_sim.sets[j] + (_sim.sets[j] * _model.factors_perturb[j])) * _model.factors_gradient[j][k]);
        }
    }
    return max;
}

void fill_position(){
    int current_pos =0;
    int delay;
    for (int i =0; i <= NUM_SPECIES; i++){
        delay = calc_delay();
        _delay_size[i]= delay;
        _position[i] = current_pos;
        current_pos += delay * _sim.cells_total;
    }
}

void initialize(){
    sum = 0
    for (int i = 0; i <= NUM_SPECIES; i++){
        sum += calc_delay();
    }
    _width = _sim.cells_total;
    _total_length = sum * _sim.cells_total;
    dealloc_array();
    _cuda=false;
    allocate_array();
    reset();
}

