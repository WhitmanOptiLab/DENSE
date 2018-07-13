#ifndef SIM_SET_HPP
#define SIM_SET_HPP

#include "utility/common_utils.hpp"
#include "core/parameter_set.hpp"
#include "core/model.hpp"
#include "cell_param.hpp"
#include "core/reaction.hpp"
#include "base.hpp"
#include "stoch/stoch.hpp"
#include "determ/determ.hpp"
#include "io/csvr.hpp"
#include "utility/style.hpp"

#include <vector>
#include <array>
#include <iostream>

/* simulation contains simulation data, partially taken from input_params and partially derived from other information
	notes:
	todo:
 */


namespace dense {

class Simulation_Set {

  public:

    std::vector<Simulation*> _sim_set;

    Simulation_Set(std::vector<Parameter_Set> parameter_sets,
      Real** gradient_factors,
      Real* perturbation_factors,
      int cell_total, int total_width,
      Real step_size, int seed)
    {
      // For each set, load data to _ps and _sim_set
      reserve(parameter_sets.size());
      for (Parameter_Set const& parameter_set : parameter_sets) {
          // When init'ing a sim_set<sim_base>, have step_size be = to 0.0 so that sim_set can emplace_back correctly
          if (step_size == 0.0) {
            emplace<Stochastic_Simulation>(
              parameter_set, perturbation_factors, gradient_factors, cell_total, total_width, seed);
          } else {
            emplace<Deterministic_Simulation>(
              parameter_set, perturbation_factors, gradient_factors, cell_total, total_width, step_size);
          }
      }
    }

    Simulation_Set() = default;

    void reserve(std::ptrdiff_t new_capacity) {
      _sim_set.reserve(new_capacity);
    }

    template <typename T, typename... Args>
    void emplace(Args&&... args) {
      _sim_set.push_back(new T(std::forward<Args>(args)...));
    }

    void simulate_sets(Real total_time, Real analysis_interval) {
        for (auto & set : _sim_set) {
            set->simulate(total_time, analysis_interval);
            set->finalize();
        }
    }

    Natural size() const {
      return _sim_set.size();
    }

};

}

#endif
