#ifndef SIM_DETERM_TRAP_HPP
#define SIM_DETERM_TRAP_HPP

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

class Trapezoid_Simulation : public Simulation, public Numerical_Integration {
  private:
    double _prev_rates[NUM_SPECIES];
    bool _first_point_calculated = false;
    bool _second_point_calculated = false;

 public:
  using Context = dense::Context<Trapezoid_Simulation>;
  Trapezoid_Simulation(const Parameter_Set& ps, Real* pnFactorsPert, Real** pnFactorsGrad,
                    Minutes step_size, std::vector<Real> conc, NGraph::Graph adj_graph);

  CUDA_AGNOSTIC
  void update_concentrations(dense::Natural cell, SpecieRates const& rates) override;

  CUDA_AGNOSTIC
  SpecieRates calculate_concentrations(dense::Natural cell);

  void step() override;
    
  virtual ~Trapezoid_Simulation() = default;
    
  Trapezoid_Simulation(Trapezoid_Simulation&&) = default;
    
  Trapezoid_Simulation & operator= (Trapezoid_Simulation&&) = default;

  CUDA_AGNOSTIC
  Minutes age_by (Minutes duration) override;


  dense::Real calculate_neighbor_average(dense::Natural cell, specie_id species, dense::Natural delay = 0) const override {
    Real sum = 0.0L;
    for (dense::Natural i = 0; i < neighbor_count_by_cell_[cell]; i++) {
        sum += _baby_cl.row_at(species, -delay)[neighbors_by_cell_[cell][i]];
    }
    Real avg = sum / neighbor_count_by_cell_[cell];
    return avg;
  }
};

}
#endif
