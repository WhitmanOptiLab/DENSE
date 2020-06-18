#ifndef SIM_DETERM_DETERM_HPP
#define SIM_DETERM_DETERM_HPP

#include "num_sim.hpp"
#include "sim/base.hpp"
#include "core/parameter_set.hpp"
#include "core/model.hpp"
#include "core/specie.hpp"
#include "sim/cell_param.hpp"
#include "core/reaction.hpp"
#include "baby_cl.hpp"

/* simulation contains simulation data, partially taken from input_params and partially derived from other information
 */

 namespace dense {

typedef cell_param<NUM_DELAY_REACTIONS, int> IntDelays;

class Deterministic_Simulation : public Simulation, public Numerical_Integration {

 public:
  using Context = dense::Context<Deterministic_Simulation>;
  Deterministic_Simulation(const Parameter_Set& ps, Real* pnFactorsPert, Real** pnFactorsGrad,
                    Minutes step_size, std::vector<Real> conc, NGraph::Graph adj_graph, dense::Natural num_grow_cell = 0);
  CUDA_AGNOSTIC
  void update_concentrations(dense::Natural cell, SpecieRates const& rates) override;
  CUDA_AGNOSTIC
  SpecieRates calculate_concentrations(dense::Natural cell);
  void step() override;
  CUDA_AGNOSTIC
  Minutes age_by (Minutes duration) override;

  dense::Real calculate_neighbor_average(dense::Natural cell, specie_id species, dense::Natural delay = 0) const override {
    // Average the given cell's neighbors' concentrations
    Real sum = 0.0L;
    for (dense::Natural i = 0; i < neighbor_count_by_cell_[cell]; i++) {
        sum += _baby_cl.row_at(species, -delay)[neighbors_by_cell_[cell][i]];
    }
    Real avg = sum / neighbor_count_by_cell_[cell];
    return std::isnan(avg) ? 0 : avg;
  }

  //add_cell: takes two cells in virtual id form and makes new cell from the parent cells history
  void add_cell(Natural cell, Natural parent = 0){
    Natural cell_index = find_id(cell); //new_index is the physical id for the virtual cell
    Natural parent_index = find_id(parent); //parent_index is the physical id for the parent virtual cell
    add_cell_base(cell);
    for (int specie = 0; specie < NUM_SPECIES; specie++) {
      int specie_size = _baby_cl.get_species_size(specie);
      for (int time = 0; time < specie_size; time++){
        _baby_cl.row_at(specie,time)[cell_index] = _baby_cl.row_at(specie,time)[parent_index];
      }
    }
  }
  
  //remove_cellL: takes a virtual cell and removes it from the simulation
  void remove_cell(Natural cell){
    remove_cell_base(cell);
  }
};

}
#endif
