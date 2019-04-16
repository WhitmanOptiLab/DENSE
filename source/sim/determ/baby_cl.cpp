#include "baby_cl.hpp"
#include "determ.hpp"
#include <iostream>

dense::baby_cl::baby_cl(Deterministic_Simulation& sim)
: _width{sim.cell_count()} {
    unsigned sum = 0;
    for (int i = 0; i < NUM_SPECIES; i++) {
        _specie_size[i] = (sim.max_delays[i] / sim._step_size) + sim._num_history_steps;
        _position[i] = sum * _width;
        sum += _specie_size[i];
    }
    _total_length = sum * _width;
    _array = new Real[_total_length]();
}