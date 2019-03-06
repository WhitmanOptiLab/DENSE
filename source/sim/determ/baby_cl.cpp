#include "baby_cl.hpp"
#include "determ.hpp"

void dense::baby_cl::initialize(){
    unsigned sum = 0;
    for (int i = 0; i < NUM_SPECIES; i++) {
        _specie_size[i] = (_sim.max_delays[i] / _sim._step_size) + _sim._num_history_steps;
        _position[i] = sum * _sim.cell_count();
        sum += _specie_size[i];
    }

    _width = _sim.cell_count();

    _total_length = sum * _sim.cell_count();

    dealloc_array();
    allocate_array();
    reset();
}
