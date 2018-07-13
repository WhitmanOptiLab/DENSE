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
 There should be only one instance of simulation at any time.
	todo:
 */


namespace dense {

class Simulation_Set {

  public:
    std::vector<Parameter_Set> const _ps;
    std::vector<Simulation*> _sim_set;
    Real total_time, analysis_interval;
    Real* factors_pert;
    Real** factors_grad;

    Simulation_Set(std::vector<Parameter_Set> params,
      Real** gradient_factors,
      Real* perturbation_factors,
      int cell_total, int total_width,
      Real step_size, Real analysis_interval,
      Real sim_time, int seed) :
        _ps(params),
        total_time{sim_time},
        analysis_interval{analysis_interval},
        factors_pert{perturbation_factors},
        factors_grad{gradient_factors}
    {
            _sim_set.reserve(_ps.size());

            // For each set, load data to _ps and _sim_set
            for (Parameter_Set const& parameter_set : _ps) {
                // When init'ing a sim_set<sim_base>, have step_size be = to 0.0 so that sim_set can emplace_back correctly
                if (step_size == 0.0) {
                    _sim_set.push_back(
                            new Stochastic_Simulation(parameter_set, factors_pert,
                                factors_grad, cell_total, total_width, seed));
                } else {
                    _sim_set.push_back(
                            new Deterministic_Simulation(parameter_set, factors_pert,
                                factors_grad, cell_total, total_width, step_size));
                }
            }
    }

    void simulate_sets() {
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
