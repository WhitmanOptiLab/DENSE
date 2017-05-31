#include "baby_cl.hpp"
#include "simulation.hpp"
using namespace std;

/*
RATETYPE baby_cl::calc_delay(int relatedReactions[]){
    int max;

    for (int i = 0; i <= sizeof(&relatedReactions); i++) {
        int j = relatedReactions[i];

        for (int k = 0; k < _sim.width_total; k++) {
            // Calculate the minimum delay, accounting for the maximum allowable perturbation and gradients
            max = MAX(max, (_sim._parameter_set._delay_sets[j] + (_sim._parameter_set._delay_sets[j] * _sim._model.factors_perturb[j])) * _sim._model.factors_gradient[j][k]);
        }
    }
    return max;
}
*/

/*
void baby_cl::fill_position(){
    for (int i =0; i <= NUM_SPECIES; i++){
    }
}
*/

void baby_cl::initialize(){
#ifndef TEST_STRUCT
    int sum =0;
    int delay =0;
    int specie_size =0;
    int current_pos =0;
#endif
    
    for (int i = 0; i < NUM_SPECIES; i++){
#ifndef TEST_STRUCT
        delay = _sim.max_delays[i];
        specie_size = delay + _sim._num_history_steps;
        sum += specie_size;
        _specie_size[i]= specie_size;
        //cout<< specie_size<<endl;
        _position[i] = current_pos;
        current_pos += specie_size * _sim._cells_total;
#else
        _sim.max_delays[i] > _max_delay ? _max_delay = _sim.max_delays[i] : 0;
#endif
    }
    
    _width = _sim._cells_total;
    
#ifndef TEST_STRUCT
    _total_length = sum * _sim._cells_total;
#else
    _max_delay += _sim._num_history_steps;
    _total_length = _max_delay * NUM_SPECIES * _sim._cells_total;
#endif

    cout<<"total_length:" <<_total_length<<endl;
    dealloc_array();
    allocate_array();
    reset();
}



CPUGPU_FUNC
baby_cl::timespan baby_cl::operator[](int i){
#ifndef TEST_STRUCT
    return timespan(_array+_position[i], _width, _specie_size[i]);
#else
    return timespan(_array+(i*_sim._cells_total), _width, _max_delay);
#endif
}

CPUGPU_FUNC
const baby_cl::timespan baby_cl::operator[](int i) const{
#ifndef TEST_STRUCT
    return timespan(_array+_position[i], _width, _specie_size[i]);
#else
    return timespan(_array+(i*_sim._cells_total), 0, 0);
#endif
}
