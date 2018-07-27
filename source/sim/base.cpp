#include <cmath>
#include "base.hpp"
#include "cell_param.hpp"
#include "model_impl.hpp"
#include <limits>
#include <iostream>
#include <bitset>
#include <algorithm>

template<int N, class T>
void dense::cell_param<N,T>::initialize_params(Parameter_Set const& ps, const Real normfactor, Real* factors_perturb, Real** factors_gradient) {
//    initialize();
    auto& self = *this;
    for (int i = 0; i < N; i++) {
      auto begin = self[i], end = begin + cell_count_;
      std::fill(begin, end, ps.data()[i] / normfactor);
      if (Real factor = factors_perturb ? factors_perturb[i] : 0.0) {
        std::transform(begin, end, begin, [=](Real param) {
          return param * random_perturbation(factor);
        });
      }
    }
    if (factors_gradient) { // If at least one rate has a gradient
        for (int i = 0; i < N; i++) {
          if (factors_gradient[i]) { // If this rate has a gradient
                // Iterate through every cell
                for (int k = 0; k < cell_count_; ++k) {
                    // Calculate the cell's index relative to the active start
                    int gradient_index = simulation_width_ - k % simulation_width_;
                    self[i][k] *= factors_gradient[i][gradient_index];
                }
            }
        }
    }
}

dense::Simulation::Simulation(Parameter_Set ps, int cells_total, int width_total, Real* factors_perturb, Real** factors_gradient) :
    _width_total(width_total), _cells_total(cells_total),
    _neighbors(new CUDA_Array<int, 6>[cells_total]), _parameter_set(std::move(ps)),
    _cellParams(width_total, cells_total), _numNeighbors(new dense::Natural[cells_total])
  {
    calc_max_delays(factors_perturb, factors_gradient);
    calc_neighbor_2d();
    _cellParams.initialize_params(_parameter_set, 1.0, factors_perturb, factors_gradient);
  }

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

void dense::Simulation::calc_max_delays(Real* factors_perturb, Real** factors_gradient) {
  for (int s = 0; s < NUM_SPECIES; s++) {
    max_delays[s] = 0.0;
  }

  std::vector<specie_id> rate_terms[NUM_REACTIONS];

  class DummyContext {
      public:
        DummyContext(std::vector<specie_id>& deps_to_fill) :
            deps(deps_to_fill) {};
        Real getCon(specie_id species, int = 0) const {
            std::ptrdiff_t sp = static_cast<std::underlying_type<Species>::type>(species);
            deps_bitset.set(sp);
            return 0.0;
        };
        Real getCon(specie_id species){
            std::ptrdiff_t sp = static_cast<std::underlying_type<Species>::type>(species);
            deps_bitset.set(sp);
            return 0.0;
        };
        Real getRate(reaction_id) const { return 0.0; };
        Real getDelay(delay_reaction_id) const { return 0.0; };
        Real getCritVal(critspecie_id) const { return 0.0; };
        Real calculateNeighborAvg(specie_id species, int = 0) const {
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
