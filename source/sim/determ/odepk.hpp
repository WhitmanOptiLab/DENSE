#ifndef SIM_DETERM_AVG_HPP
#define SIM_DETERM_AVG_HPP

#include "num_sim.hpp"
#include "sim/base.hpp"
#include "core/parameter_set.hpp"
#include "core/model.hpp"
#include "core/specie.hpp"
#include "sim/cell_param.hpp"
#include "core/reaction.hpp"
#include "baby_cl.hpp"

namespace dense {

  typedef cell_param<NUM_DELAY_REACTIONS, int> IntDelays;

  class ODEpack_Simulation : public Simulation, public Numerical_Integration {
  private:
    bool _first_point_calculated = false;
    bool _second_point_calculated = false;
    bool _third_point_calculated = false;
    std::vector<vector<double>> _n_minus_1_rates;
    std::vector<vector<double>> _n_minus_2_rates;
    std::vector<vector<double>> _n_minus_3_rates;
  public:
    using Context = dense::Context<ODEpack_Simulation>;
    ODEpack_Simulation(const Parameter_Set& ps, Real* pnFactorsPert, Real** pnFactorsGrad,
                    Minutes step_size, std::vector<Real> conc, NGraph::Graph adj_graph);

    CUDA_AGNOSTIC
    SpecieRates calculate_concentrations(dense::Natural cell);

    void step() override;

    virtual ~ODEpack_Simulation() = default;

    ODEpack_Simulation(ODEpack_Simulation&&) = default;

    ODEpack_Simulation & operator= (ODEpack_Simulation&&) = default;

    dense::Real calculate_neighbor_average(dense::Natural cell, specie_id species, dense::Natural delay = 0) const override {
      // Average the given cell's neighbors' concentrations
      Real sum = 0.0L;
      for (dense::Natural i = 0; i < neighbor_count_by_cell_[cell]; i++) {
          sum += _baby_cl.row_at(species, -delay)[neighbors_by_cell_[cell][i]];
      }
      Real avg = sum / neighbor_count_by_cell_[cell];
      return avg;
    }

    CUDA_AGNOSTIC
    Minutes age_by (Minutes duration) override;

    CUDA_AGNOSTIC
    void update_concentrations(dense::Natural cell, SpecieRates const& rates) override;
  };
}

#endif
