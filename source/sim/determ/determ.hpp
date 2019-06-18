#ifndef SIM_DETERM_DETERM_HPP
#define SIM_DETERM_DETERM_HPP

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

class Deterministic_Simulation : public Simulation {

 public:

    using Context = dense::Context<Deterministic_Simulation>;
    using SpecieRates = CUDA_Array<Real, NUM_SPECIES>;

  // PSM stands for Presomitic Mesoderm (growth region of embryo)
  IntDelays _intDelays;

  // Sizes
  Real _step_size; // The step size in minutes
  //int steps_total; // The number of time steps to simulate (total time / step size)
  //int steps_split; // The number of time steps it takes for cells to split
  //int steps_til_growth; // The number of time steps to wait before allowing cells to grow into the anterior PSM
  //bool no_growth; // Whether or not the simulation should rerun with growth

  // Granularities
  //int _big_gran; // The granularity in time steps with which to analyze and store data
  //int small_gran; // The granularit in time steps with which to simulate data

  // Neighbors and boundaries
  //array2D<int> neighbors; // An array of neighbor indices for each cell position used in 2D simulations (2-cell and 1D calculate these on the fly)
  //int active_start; // The start of the active portion of the PSM
  //int active_end; // The end of the active portion of the PSM

  // PSM section and section-specific times
  //int section; // Posterior or anterior (sec_post or sec_ant)
  //int time_start; // The start time (in time steps) of the current simulation
  //int time_end; // The end time (in time steps) of the current simulation
  //int time_baby; // Time 0 for baby_cl at the end of a simulation

  // Mutants and condition scores
  //int num_active_mutants; // The number of mutants to simulate for each parameter set
  //double max_scores[NUM_SECTIONS]; // The maximum score possible for all mutants for each testing section
  //double max_score_all; // The maximum score possible for all mutants for all testing sections

  int _j;
  dense::Natural _num_history_steps; // how many steps in history are needed for this numerical method
private:
  baby_cl _baby_cl;
public:
  Deterministic_Simulation(const Parameter_Set& ps, Real* pnFactorsPert, Real** pnFactorsGrad, int tissue_width,
                    Minutes step_size, std::vector<Real> conc, NGraph::Graph adj_graph);

  CUDA_AGNOSTIC
  void update_concentrations(dense::Natural cell, SpecieRates const& rates);

  CUDA_AGNOSTIC
  SpecieRates calculate_concentrations(dense::Natural cell);

  void step();

  Real get_concentration(dense::Natural cell, specie_id species) const {
    return get_concentration(cell, species, 1);
  }

  Real get_concentration(dense::Natural cell, specie_id species, dense::Natural delay) const {
    return _baby_cl.row_at(species, 1 - delay)[cell];
  }

  dense::Real calculate_neighbor_average(dense::Natural cell, specie_id species, dense::Natural delay = 0) const {
    // Average the given cell's neighbors' concentrations
    Real sum = 0.0L;
    for (dense::Natural i = 0; i < neighbor_count_by_cell_[cell]; i++) {
        sum += _baby_cl.row_at(species, -delay)[neighbors_by_cell_[cell][i]];
    }
    Real avg = sum / neighbor_count_by_cell_[cell];
    return avg;
  }

  CUDA_AGNOSTIC
  Minutes age_by (Minutes duration);

};

}
#endif
