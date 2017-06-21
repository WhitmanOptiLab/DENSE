#include "cell_param.hpp"
#include "simulation_base.hpp"

#include <iostream>


using namespace std;

template<int N, class T>
void cell_param<N,T>::update_rates(const RATETYPE param_data[], const RATETYPE normfactor){
//    initialize();
    if (_sim._model._using_perturb){
        for (int i = 0; i < N; i++) {
            if (_sim._model.factors_perturb[i] == 0) { // If the current rate has no perturbation factor then set every cell's rate to the base rate
                for (int j = 0; j < _sim._cells_total; j++) {
                    //double rnum;
                    //rnum=0.082;
                    _array[_width*i+j] = param_data[i]/normfactor;
                }
            } else { // If the current rate has a perturbation factor then set every cell's rate to a randomly perturbed positive or negative variation of the base with a maximum perturbation up to the rate's perturbation factor
                for (int j = 0; j < _sim._cells_total; j++) {
                    _array[_width*i+j] = param_data[i] * random_perturbation(_sim._model.factors_perturb[i]) / normfactor;
                }
            }
        }
    } else {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < _sim._cells_total; j++) {
                _array[_width*i+j] =param_data[i] / normfactor;
            }
        }
    }
    if (_sim._model._using_gradients) { // If at least one rate has a gradient
        for (int i = 0; i < N; i++) {
            if (_sim._model._has_gradient[i]) { // If this rate has a gradient
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
                    _array[_width*i+k] *= _sim._model.factors_gradient[i][gradient_index];
                }
            }
        }
    }
}

//Dummy function to force generation of update_rates for all the simulation types
void __genUpdateRates(simulation_base& s) {
  Rates r(s, 1);
  r.update_rates(NULL);
  Delays d(s, 1);
  d.update_rates(NULL);
  CritValues c(s, 1);
  c.update_rates(NULL);
}

template<int N, class T>
void cell_param<N,T>:: initialize(){
    _width = _sim._cells_total;
    dealloc_array();
    allocate_array();
}
