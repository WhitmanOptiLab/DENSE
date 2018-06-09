#include "baby_cl_cuda.hpp"
#include "determ_cuda.hpp"

void baby_cl_cuda::initialize(){
    int sum =0;
    int delay =0;
    int specie_size =0;
    int current_pos =0;

    for (int i = 0; i < NUM_SPECIES; i++){
        delay = _sim.max_delays[i] / _sim._step_size;
        specie_size = delay + _sim._num_history_steps;
        sum += specie_size;
        _specie_size[i]= specie_size;
        _position[i] = current_pos;
        current_pos += specie_size * _sim._cells_total;
    }
    _width = _sim._cells_total;
    _total_length = sum * _sim._cells_total;
    std::cout << "total_length:" << _total_length << '\n';
    dealloc_array();
    allocate_array();
    reset();
}
