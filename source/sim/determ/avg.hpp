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

  class Average_Simulation : public Simulation, public Numerical_Integration {
  private:
    int _curr_coeff = 1;
    bool _first_point_calculated = false;
    bool _second_point_calculated = false;
    double _prev_rates[NUM_SPECIES];
    float _simpson_value(float curr_rate, dense::Natural cell, int index);
    float _euler_value(float curr_rate, dense::Natural cell, int index);
    float _trapezoid_value(float curr_rate, dense::Natural cell, int index);
  public:
    using Context = dense::Context<Average_Simulation>;
    Average_Simulation(const Parameter_Set& ps, Real* pnFactorsPert, Real** pnFactorsGrad,
                    Minutes step_size, std::vector<Real> conc, NGraph::Graph adj_graph);

    CUDA_AGNOSTIC
    SpecieRates calculate_concentrations(dense::Natural cell);

    void step() override;

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
