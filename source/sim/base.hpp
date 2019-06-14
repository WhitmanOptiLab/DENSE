#ifndef SIM_BASE_HPP
#define SIM_BASE_HPP

#include "utility/common_utils.hpp"
#include "utility/cuda.hpp"
#include "core/parameter_set.hpp"
#include "core/model.hpp"
#include "core/specie.hpp"
#include "cell_param.hpp"
#include "core/reaction.hpp"
#include "ngraph/ngraph_components.hpp"

#include <vector>
#include <iostream>
#include <initializer_list>
#include <algorithm>
#include <chrono>
#include <type_traits>
#include <utility>


# if defined __CUDA_ARCH__
using Minutes = Real;
# else
using Minutes = std::chrono::duration<Real, std::chrono::minutes::period>;
# endif

# if defined __cpp_concepts
template <typename T>
concept bool Simulation_Concept() {
  return requires(T& simulation, T const& simulation_const) {
    { simulation_const.age() } noexcept -> Minutes;
    { simulation.age_by(Minutes{}) } -> Minutes;
    { simulation_const.cell_count() } noexcept -> Natural;
    { simulation_const.get_concentration(Natural{}, specie_id{}, Natural{}) } -> Real;
    { simulation_const.get_concentration(Natural{}, specie_id{}) } -> Real;
    { simulation_const.calculate_neighbor_average(Natural{}, specie_id{}, Natural{}) } -> Real;
  };
}
# endif


namespace dense {

class Simulation;

template <typename Simulation_T>
class Context {

    static_assert(std::is_base_of<dense::Simulation, Simulation_T>(),
      "Class template <typename T> dense::Context requires std::is_base_of<Simulation, T>()");

  public:

    CUDA_AGNOSTIC
    Context(Simulation_T & owner, dense::Natural cell = 0)
    : owner_{std::addressof(owner)}, cell_{cell} {
    }

    CUDA_AGNOSTIC
    Real getCon(specie_id sp) const;

    CUDA_AGNOSTIC
    Real getCon(specie_id sp, int delay) const;

    CUDA_AGNOSTIC
    Real calculateNeighborAvg(specie_id sp, int delay) const;

    CUDA_AGNOSTIC
    Real getCritVal(critspecie_id rcritsp) const;

    CUDA_AGNOSTIC
    Real getRate(reaction_id reaction) const;

    CUDA_AGNOSTIC
    Real getDelay(delay_reaction_id delay_reaction) const;

  private:

    Simulation_T * owner_;

    Natural cell_;

};

/* simulation contains simulation data, partially taken from input_params and partially derived from other information
 */
/* SIMULATION_BASE
 * superclass for simulation_determ and simulation_stoch
 * inherits from Observable, can be observed by Observer object
*/
///
///
class Simulation {

  using This = Simulation;

  public:

    /// (Deleted) Copy-construct a simulation.
    Simulation (This const&) = delete;

    /// Move-construct a simulation.
    CUDA_AGNOSTIC
    Simulation (This&&) noexcept;

    /// (Deleted) Copy-assign to a simulation.
    This& operator= (This const&) = delete;

    /// Move-assign to a simulation.
    CUDA_AGNOSTIC
    This& operator= (This&&);

    /// Determine the current number of cells comprising a simulation.
    CUDA_AGNOSTIC
    Natural cell_count () const noexcept;

    /// Determine the age of a simulation in virtual minutes.
    CUDA_AGNOSTIC
    Minutes age () const noexcept;

    /// Age (advance) a simulation by the specified number of virtual minutes.
    CUDA_AGNOSTIC
    Minutes age_by (Minutes duration) noexcept;

    CUDA_AGNOSTIC
    Natural& cell_count () noexcept;

  protected:

    CUDA_AGNOSTIC
    Simulation () = default;

    /*
     * CONSTRUCTOR
     * arg "ps": assiged to "_parameter_set", used to access user-inputted rate constants, delay times, and crit values
     * arg "cells_total": the maximum amount of cells to simulate for (initial count for non-growing tissues)
     * arg "width_total": the circumference of the tube, in cells
    */
    Simulation(Parameter_Set parameter_set, NGraph::Graph* adj_graph, Real* perturbation_factors = nullptr, Real** gradient_factors = nullptr) noexcept;

    CUDA_AGNOSTIC
    ~Simulation () noexcept;


  private:

    void calc_max_delays(Real*, Real**);
  
  
    /*
     * CALC_NEIGHBOR_2D
     * populates the data structure "_neighbors" with cell indices of neighbors
     * loads graph from "adjacency_graph", either user specified or default graph from graph_constructor()
    */
    CUDA_AGNOSTIC
    void calc_neighbor_2d() noexcept {
      for ( Graph::const_iterator p = adjacency_graph->begin(); p != adjacency_graph->end(); p++){
          Graph::vertex_set neigh = Graph::out_neighbors(p);
          std::vector<Natural>* neighbors = new std::vector<Natural> [neighbors->size()];
          auto index = adjacency_graph->node(p);
          for ( Graph::vertex_set::const_iterator cell = neigh.begin(); cell != neigh.end(); cell++){
            neighbors->push_back(*cell);
          }
          neighbors_by_cell_[index] = *neighbors;
          neighbor_count_by_cell_[index] = neighbors->size();
          delete[] neighbors;
        }
    }

  private:

    Real age_ = {};
    //Natural circumference_ = {};
    Natural cell_count_ = {};
    Parameter_Set parameter_set_ = {};
    int index;
    NGraph::Graph* adjacency_graph;

  protected:

    std::vector<std::vector<Natural>> neighbors_by_cell_ = {};
    Natural* neighbor_count_by_cell_ = {};

  public:

    Real max_delays[NUM_SPECIES] = {};  // The maximum number of time steps that each specie might be accessed in the past
    cell_param<NUM_PARAMS> cell_parameters_ = {};

  // Sizes
  // Real* factors_perturb;
  // Real** factors_gradient;
  //Natural _width_total = {}; // The maximum width in cells of the PSM
  //dense::Natural circumf; // The current width in cells
  //int _width_initial; // The width in cells of the PSM before anterior growth
  //int _width_current; // The width in cells of the PSM at the current time step
  //int height; // The height in cells of the PSM
  //int cells; // The number of cells in the simulation

  // Times and timing
  //int steps_total; // The number of time steps to simulate (total time / step size)
  //int steps_split; // The number of time steps it takes for cells to split
  //int steps_til_growth; // The number of time steps to wait before allowing cells to grow into the anterior PSM
  //bool no_growth; // Whether or not the simulation should rerun with growth

  // Neighbors and boundaries
  //array2D<int> neighbors; // An array of neighbor indices for each cell position used in 2D simulations (2-cell and 1D calculate these on the fly)
  //int active_start; // The start of the active portion of the PSM
  //int active_end; // The end of the active portion of the PSM

  // PSM section and section-specific times
  //int section; // Posterior or anterior (sec_post or sec_ant)
  //int time_start; // The start time (in time steps) of the current simulation
  //int time_end; // The end time (in time steps) of the current simulation

  // Mutants and condition scores
  //int num_active_mutants; // The number of mutants to simulate for each parameter set
  //double max_scores[NUM_SECTIONS]; // The maximum score possible for all mutants for each testing section
  //double max_score_all; // The maximum score possible for all mutants for all testing sections

  //CPUGPU_TempArray<int,NUM_SPECIES> _baby_j;
  //int* _delay_size;
  //int* _time_prev;
  //double* _sets;
  //int _NEIGHBORS_2D;

};


CUDA_AGNOSTIC
inline Simulation::Simulation (Simulation&&) noexcept = default;

CUDA_AGNOSTIC
inline Simulation& Simulation::operator= (Simulation&&) = default;

CUDA_AGNOSTIC
inline Natural Simulation::cell_count() const noexcept {
  return cell_count_;
}

CUDA_AGNOSTIC
inline Natural& Simulation::cell_count() noexcept {
  return cell_count_;
}

CUDA_AGNOSTIC
inline Minutes Simulation::age () const noexcept {
  return Minutes{ age_ };
}

CUDA_AGNOSTIC
inline Minutes Simulation::age_by (Minutes duration) noexcept {
  return Minutes{ age_ += duration / Minutes{1} };
}

CUDA_AGNOSTIC
inline Simulation::~Simulation () noexcept = default;

}

template <typename T>
Real dense::Context<T>::getCritVal(critspecie_id rcritsp) const {
  return owner_->cell_parameters_[NUM_REACTIONS+NUM_DELAY_REACTIONS+rcritsp][cell_];
}

template <typename T>
Real dense::Context<T>::getRate(reaction_id reaction) const {
    return owner_->cell_parameters_[reaction][cell_];
}

template <typename T>
Real dense::Context<T>::getDelay(delay_reaction_id delay_reaction) const {
  return owner_->cell_parameters_[NUM_REACTIONS+delay_reaction][cell_];
}

template <typename T>
dense::Real dense::Context<T>::getCon(specie_id sp) const {
  return owner_->get_concentration(cell_, sp);
}

template <typename T>
dense::Real dense::Context<T>::getCon(specie_id sp, int delay) const {
  return owner_->get_concentration(cell_, sp, delay);
}

template <typename T>
dense::Real dense::Context<T>::calculateNeighborAvg(specie_id sp, int delay) const {
  return owner_->calculate_neighbor_average(cell_, sp, delay);
}


#endif
