#include <cmath>
#include "base.hpp"
#include "cell_param.hpp"
#include "model_impl.hpp"
#include <limits>
#include <iostream>
#include <bitset>

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


void Simulation::initialize(){
    calc_max_delays();
    calc_neighbor_2d();
    _cellParams.initialize_params(_parameter_set);
}

void Simulation::calc_max_delays() {
  for (int s = 0; s < NUM_SPECIES; s++) {
    max_delays[s] = 0.0;
  }

  std::vector<specie_id> rate_terms[NUM_REACTIONS];

  class DummyContext {
      public:
        DummyContext(std::vector<specie_id>& deps_to_fill) :
            deps(deps_to_fill) {};
        Real getCon(specie_id species, int delay = 0) const {
            std::ptrdiff_t sp = static_cast<std::underlying_type<Species>::type>(species);
            deps_bitset.set(sp);
            return 0.0;
        };
        Real getCon(specie_id species){
            std::ptrdiff_t sp = static_cast<std::underlying_type<Species>::type>(species);
            deps_bitset.set(sp);
            return 0.0;
        };
        Real getRate(reaction_id rid) const { return 0.0; };
        Real getDelay(delay_reaction_id rid) const { return 0.0; };
        Real getCritVal(critspecie_id crit) const { return 0.0; };
        Real calculateNeighborAvg(specie_id species, int delay = 0) const {
            std::ptrdiff_t sp = static_cast<std::underlying_type<Species>::type>(species);
            deps_bitset.set(sp);
            return 0.0;
        };

        ~DummyContext () {
          for (std::size_t i = 0; i < deps_bitset.size() && deps.size() < deps_bitset.count(); ++i) {
            if (deps_bitset.test(i)) {
              deps.push_back(static_cast<Species>(i));
            }
          }
        }
      private:
        std::vector<specie_id>& deps;
        mutable std::bitset<Species::size> deps_bitset{};
  };


    #define REACTION(name) \
    dense::model::reaction_##name.active_rate( DummyContext (rate_terms[name]));
    #include "reactions_list.hpp"
    #undef REACTION

  //for each reaction
  //  for each input
  //    accumulate delay into specie
  //  for each factor
  //    accumulate delay into specie
 std::vector<specie_id> delts;
#define REACTION(name)
#define DELAY_REACTION(name) \
  delts = rate_terms[name]; \
  Real max_gradient_##name = 1.0; \
  if (factors_gradient) \
  { \
    for (dense::Natural k = 0; k < _width_total && factors_gradient[name]; k++) { \
      max_gradient_##name = std::max<Real>(factors_gradient[ name ][k], max_gradient_##name); \
    } \
  } \
  \
  Real pert_##name = 0.0; \
  if (factors_perturb) \
  { \
    pert_##name = factors_perturb[name]; \
  } \
  \
  for (auto factor : delts) { \
    Real& sp_max_delay = max_delays[factor]; \
    sp_max_delay = std::max<Real>((_parameter_set.getDelay(dreact_##name) * max_gradient_##name * (1.0 + pert_##name) ), sp_max_delay); \
  }
#include "reactions_list.hpp"
#undef REACTION
#undef DELAY_REACTION
}
