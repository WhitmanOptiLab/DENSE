#include "baby_cl.hpp"
#include "simulation.hpp"
using namespace std;

void baby_cl::initialize(){
    int sum =0;
    int delay =0;
    int specie_size =0;
    int current_pos =0;
    
    for (int i = 0; i < NUM_SPECIES; i++){
        delay = _sim.max_delays[i];
        specie_size = delay + _sim._num_history_steps;
        sum += specie_size;
        _specie_size[i]= specie_size;
        //cout<< specie_size<<endl;
        _position[i] = current_pos;
        current_pos += specie_size * _sim._cells_total;
    }
    
    _width = _sim._cells_total;
    
    _total_length = sum * _sim._cells_total;

    cout<<"  Length: " <<_total_length<<endl;
    dealloc_array();
    allocate_array();
    reset();
}

