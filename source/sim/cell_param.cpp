#include "cell_param.hpp"
#include "base.hpp"

#include <iostream>




template<int N, class T>
void cell_param<N,T>::initialize_params(param_set const& ps, const RATETYPE normfactor){
//    initialize();
    if (_sim.factors_perturb){
        for (int i = 0; i < N; i++) {
            if (_sim.factors_perturb[i] == 0) { // If the current rate has no perturbation factor then set every cell's rate to the base rate
                for (int j = 0; j < _sim._cells_total; j++) {
                    //double rnum;
                    //rnum=0.082;
                    _array[_width*i+j] = ps.getArray()[i]/normfactor;
                }
            } else { // If the current rate has a perturbation factor then set every cell's rate to a randomly perturbed positive or negative variation of the base with a maximum perturbation up to the rate's perturbation factor
                for (int j = 0; j < _sim._cells_total; j++) {
                    _array[_width*i+j] = ps.getArray()[i] * random_perturbation(_sim.factors_perturb[i]) / normfactor;
                }
            }
        }
    } else {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < _sim._cells_total; j++) {
                _array[_width*i+j] = ps.getArray()[i] / normfactor;
            }
        }
    }
    if (_sim.factors_gradient) { // If at least one rate has a gradient
        for (int i = 0; i < N; i++) {
            if (_sim.factors_gradient[i]) { // If this rate has a gradient
                // Iterate through every cell
                for (int k = 0; k < _sim._cells_total; k++) {
                    // Calculate the cell's index relative to the active start
                    int col = k % _sim._width_total;
                    int gradient_index;
                    //if (col <= active_start) {
                        gradient_index = _sim._width_total - col;
                    //} else {
                    //    gradient_index = active_start + rs.width - col;
                    //}
                    
                    // Set the cell's active rate to its perturbed rate modified by its position's gradient factor
                    _array[_width*i+k] *= _sim.factors_gradient[i][gradient_index];
                }
            }
        }
    }
}

//Dummy function to force generation of update_params for all the simulation types
void __genUpdateRates(simulation_base& s) {
  cell_param<NUM_REACTIONS+NUM_DELAY_REACTIONS+NUM_CRITICAL_SPECIES, RATETYPE> r(s, 1);
  param_set ps;
  r.initialize_params(ps);
}

template<int N, class T>
void cell_param<N,T>:: initialize(){
    _width = _sim._cells_total;
    dealloc_array();
    allocate_array();
}
