#include "dense_cl.hpp"
#include "determ.hpp"
using namespace std;

void dense_cl::initialize(){
    for (int i = 0; i < NUM_SPECIES; i++){
        _sim.max_delays[i] > _max_delay ? _max_delay = _sim.max_delays[i] : 0;
    }
    
    _width = _sim._cells_total;
    
    _max_delay += _sim._num_history_steps;
    _total_length = _max_delay * NUM_SPECIES * _sim._cells_total;

    cout<<"total_length:" <<_total_length<<endl;
    dealloc_array();
    allocate_array();
    reset();
}

