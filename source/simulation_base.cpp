#include <cmath>
#include "simulation_base.hpp"
#include "cell_param.hpp"
#include "model_impl.hpp"
#include <limits>
#include <iostream>

typedef std::numeric_limits<double> dbl;
using namespace std;

/*
bool simulation_base::any_less_than_0 (baby_cl& baby_cl, int* times) {
    for (int i = 0; i <= NUM_SPECIES; i++) {
        int time = times[i];
        if (baby_cl[i][time][0] < 0) { // This checks only the first cell
            return true;
        }
    }
    return false;
}

bool simulation_base::concentrations_too_high (baby_cl& baby_cl, int* times, double max_con_thresh) {
    if (max_con_thresh != INFINITY) {
        for (int i = 0; i <= NUM_SPECIES; i++) {
            int time = times[i];
            if (baby_cl[i][time][0] > max_con_thresh) { // This checks only the first cell
                return true;
            }
        }
    }
    return false;
}
*/

void simulation_base::initialize(){
    for (int c=0; c<_cells_total; c++){
        vector<int> v;
        _neighbors.push_back(v);
    }
    calc_max_delays();
    calc_neighbor_2d();
    _delays.update_rates(_parameter_set._delay_sets);
    _rates.update_rates(_parameter_set._rates_base);
    _critValues.update_rates(_parameter_set._critical_values);
}
    
void simulation_base::calc_max_delays() {
  for (int s = 0; s < NUM_SPECIES; s++) {
    max_delays[s] = 0.0;
  }
  //for each reaction
  //  for each input
  //    accumulate delay into specie
  //  for each factor
  //    accumulate delay into specie
  //RATETYPE max_gradient_##name = 0; \
  //for (int k = 0; k < _width_total; k++) { \
  //  max_gradient_##name = std::max<int>(_model.factors_gradient[ name ][k], max_gradient_##name); \
  //} 
#define REACTION(name) 
#define DELAY_REACTION(name) \
  for (int in = 0; in < _model.reaction_##name.getNumInputs(); in++) { \
    RATETYPE& sp_max_delay = max_delays[_model.reaction_##name.getInputs()[in]]; \
    sp_max_delay = std::max<RATETYPE>(_parameter_set._delay_sets[ dreact_##name ], sp_max_delay); \
  } \
  for (int in = 0; in < _model.reaction_##name.getNumFactors(); in++) { \
    RATETYPE& sp_max_delay = max_delays[_model.reaction_##name.getFactors()[in]]; \
    sp_max_delay = std::max<RATETYPE>(_parameter_set._delay_sets[ dreact_##name ], sp_max_delay); \
  }
#include "reactions_list.hpp"
#undef REACTION
#undef DELAY_REACTION
}
