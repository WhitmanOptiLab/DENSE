#include "rates.hpp"
#include "simulation.hpp"

#include <iostream>


using namespace std;
void Rates::update_rates(){
    if (_sim._model._using_perturb){
        for (int i = 0; i < NUM_SPECIES; i++) {
            if (_sim._model.factors_perturb[i] == 0) { // If the current rate has no perturbation factor then set every cell's rate to the base rate
                for (int j = 0; j < _sim.cells_total; j++) {
                    //double rnum;
                    //rnum=0.082;
                    _array[_width*i+j] = _sim._parameter_set._rates_base[i];
                }
            } else { // If the current rate has a perturbation factor then set every cell's rate to a randomly perturbed positive or negative variation of the base with a maximum perturbation up to the rate's perturbation factor
                for (int j = 0; j < _sim.cells_total; j++) {
                    _array[_width*i+j] = _sim._parameter_set._rates_base[i] * random_perturbation(_sim._model.factors_perturb[i]);
                }
            }
        }
    }
    if (_sim._model._using_gradients) { // If at least one rate has a gradient
        for (int i = 0; i < NUM_SPECIES; i++) {
            if (_sim._model._has_gradient[i]) { // If this rate has a gradient
                // Iterate through every cell
                for (int k = 0; k < _sim.cells_total; k++) {
                    // Calculate the cell's index relative to the active start
                    int col = k % _sim.width_total;
                    int gradient_index;
                    //if (col <= active_start) {
                        gradient_index = _sim.width_total - col;
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
