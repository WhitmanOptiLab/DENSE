#include <cmath>
#include "simulation_base.hpp"
#include "cell_param.hpp"
#include "model_impl.hpp"
#include <limits>
#include <iostream>
#include <set>

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
   
  set<specie_id> rate_terms[NUM_REACTIONS];

  class DummyContext {
      public:
        DummyContext(set<specie_id>& deps_to_fill) : 
            deps(deps_to_fill) {};
        RATETYPE getCon(specie_id sp, int delay=0) const {
            deps.insert(sp);
        };
        RATETYPE getCon(specie_id sp){
            deps.insert(sp);
        };
        RATETYPE getRate(reaction_id rid) const { return 0.0; };
        RATETYPE getDelay(delay_reaction_id rid) const { return 0.0; };
        RATETYPE getCritVal(critspecie_id crit) const { return 0.0; };
        RATETYPE calculateNeighborAvg(specie_id sp, int delay=0) const { 
            deps.insert(sp);
        };
      private:
        set<specie_id>& deps;
  };


    #define REACTION(name) \
    const reaction<name>& r##name = _model.reaction_##name; \
    r##name.active_rate( DummyContext (rate_terms[name]));
    #include "reactions_list.hpp"
    #undef REACTION

  //for each reaction
  //  for each input
  //    accumulate delay into specie
  //  for each factor
  //    accumulate delay into specie
 set<specie_id> delts;
 std::set<specie_id>::iterator iter;
#define REACTION(name) 
#define DELAY_REACTION(name) \
  delts = rate_terms[name]; \
  iter = delts.begin(); \
  RATETYPE max_gradient_##name = 1.0; \
  for (int k=0; k<_width_total && _model._has_gradient[name]; k++) { \
    max_gradient_##name = std::max<RATETYPE>(_model.factors_gradient[ name ][k], max_gradient_##name); \
  } \
  for (int in = 0; in < delts.size(); in++) { \
    std::advance(iter,in); \
    RATETYPE& sp_max_delay = max_delays[*iter]; \
    sp_max_delay = std::max<RATETYPE>((_parameter_set._delay_sets[ dreact_##name ] * max_gradient_##name * (1.0 + _model.factors_perturb[ name ]) ), sp_max_delay); \
  }
#include "reactions_list.hpp"
#undef REACTION
#undef DELAY_REACTION
}
