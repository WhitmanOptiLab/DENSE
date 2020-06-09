#ifndef SIM_DETERM_NUM_SIM_HPP
#define SIM_DETERM_NUM_SIM_HPP

#include "sim/base.hpp"
#include "core/parameter_set.hpp"
#include "core/model.hpp"
#include "core/specie.hpp"
#include "sim/cell_param.hpp"
#include "core/reaction.hpp"
#include "baby_cl.hpp"

namespace dense {
  typedef cell_param<NUM_DELAY_REACTIONS, int> IntDelays;

  class Numerical_Integration {

  public:
    using SpecieRates = CUDA_Array<Real, NUM_SPECIES>;
    IntDelays _intDelays;
    Real _step_size;
    int _j;
    dense::Natural _num_history_steps; // how many steps in history are needed for this numerical method
    Numerical_Integration(int num_delay_rxn, Natural& cell_cnt, Minutes& step_size, Deterministic_Simulation& sim);
    Numerical_Integration(int num_delay_rxn, Natural& cell_cnt, Minutes& step_size, Simpson_Simulation& sim);
  protected:
    baby_cl _baby_cl;
  public:
    Real get_concentration(dense::Natural cell, specie_id species) const {
      return get_concentration(cell, species, 1);
    }

    Real get_concentration(dense::Natural cell, specie_id species, dense::Natural delay) const {
      return _baby_cl.row_at(species, 1 - delay)[cell];
    }

    CUDA_AGNOSTIC
    virtual void update_concentrations(dense::Natural cell, SpecieRates const& rates);

    virtual void step() = 0;

    CUDA_AGNOSTIC
    virtual Minutes age_by(Minutes duration) = 0;

    virtual dense::Real calculate_neighbor_average(dense::Natural cell,
                specie_id species, dense::Natural delay = 0) const = 0;

  };
}

#endif
