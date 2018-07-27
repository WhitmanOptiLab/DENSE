#ifndef SIM_BASE_HPP
#define SIM_BASE_HPP

#include "utility/common_utils.hpp"
#include "core/parameter_set.hpp"
#include "core/model.hpp"
#include "core/specie.hpp"
#include "cell_param.hpp"
#include "core/reaction.hpp"

#include <vector>
#include <iostream>
#include <initializer_list>
#include <algorithm>
#include <chrono>
#include <type_traits>

#ifdef __cpp_concepts
template <typename T>
concept bool Simulation_Concept() {
  return requires(T& simulation, T const& simulation_const) {
    { simulation.simulate_for(Real{}) };
    { simulation.simulate_for(std::chrono::duration<Real, std::chrono::minutes::period>{}) };
    { simulation_const.get_concentration(Natural{}, specie_id{}, Natural{}) } -> Real;
    { simulation_const.get_concentration(Natural{}, specie_id{}) } -> Real;
    { simulation_const.calculate_neighbor_average(Natural{}, specie_id{}, Natural{}) } -> Real;
  };
}
#endif

namespace dense {

class Simulation;

template <typename Simulation_T>
class Context {

    static_assert(std::is_base_of<dense::Simulation, Simulation_T>(),
      "Class template <typename T> dense::Context requires std::is_base_of<Simulation, T>()");

  public:

    CUDA_HOST CUDA_DEVICE
    Context(Simulation_T & owner, dense::Natural cell = 0)
    : owner_{std::addressof(owner)}, cell_{cell} {
    }

    CUDA_HOST CUDA_DEVICE
    Real getCon(specie_id sp) const;

    CUDA_HOST CUDA_DEVICE
    Real getCon(specie_id sp, int delay) const;

    CUDA_HOST CUDA_DEVICE
    Real calculateNeighborAvg(specie_id sp, int delay) const;

    CUDA_HOST CUDA_DEVICE
    Real getCritVal(critspecie_id rcritsp) const;

    CUDA_HOST CUDA_DEVICE
    Real getRate(reaction_id reaction) const;

    CUDA_HOST CUDA_DEVICE
    Real getDelay(delay_reaction_id delay_reaction) const;

  private:

    Simulation_T * owner_;

    Natural cell_;

};


using Seconds = std::chrono::duration<Real>;
using Minutes = std::chrono::duration<Real, std::chrono::minutes::period>;


/* simulation contains simulation data, partially taken from input_params and partially derived from other information
 */
/* SIMULATION_BASE
 * superclass for simulation_determ and simulation_stoch
 * inherits from Observable, can be observed by Observer object
*/
class Simulation {

  public:

    /*
     * CONSTRUCTOR
     * arg "ps": assiged to "_parameter_set", used to access user-inputted rate constants, delay times, and crit values
     * arg "cells_total": the maximum amount of cells to simulate for (initial count for non-growing tissues)
     * arg "width_total": the circumference of the tube, in cells
    */
    Simulation(Parameter_Set parameter_set, int cells_total, int width_total, Real* perturbation_factors = nullptr, Real** gradient_factors = nullptr);

    Simulation() : Simulation(Parameter_Set(), 0, 0) {}

    Minutes age() const {
      return Minutes(age_);
    }

    Minutes age_by (Minutes duration) {
      return age_by(duration.count());
    }

    Minutes age_by (Real duration_in_minutes) {
      return Minutes(age_ += duration_in_minutes);
    }

  protected:

    ~Simulation() noexcept = default;

  private:

    void calc_max_delays(Real*, Real**);

    /*
     * CALC_NEIGHBOR_2D
     * populates the data structure "_neighbors" with cell indices of neighbors
     * follows hexagonal adjacencies for an unfilled tube
    */
    CUDA_HOST CUDA_DEVICE
    void calc_neighbor_2d() noexcept {
      for (dense::Natural i = 0; i < _cells_total; ++i) {
        bool is_former_edge = i % _width_total == 0;
        bool is_latter_edge = (i + 1) % _width_total == 0;
        bool is_even = i % 2 == 0;
        auto la = (is_former_edge || !is_even) ? _width_total - 1 : -1;
        auto ra = !(is_latter_edge || is_even) ? _width_total + 1 :  1;

        auto top          = (i - _width_total      + _cells_total) % _cells_total;
        auto bottom       = (i + _width_total                    ) % _cells_total;
        auto bottom_right = (i                + ra               ) % _cells_total;
        auto top_left     = (i                + la               ) % _cells_total;
        auto top_right    = (i - _width_total + ra + _cells_total) % _cells_total;
        auto bottom_left  = (i - _width_total + la + _cells_total) % _cells_total;

        if (is_former_edge) {
          _neighbors[i] = { top, top_right, top_left, bottom_left };
          _numNeighbors[i] = 4;
        } else if (is_latter_edge) {
          _neighbors[i] = { top, top_right, bottom_right, bottom };
          _numNeighbors[i] = 4;
        } else {
          _neighbors[i] = { top, top_right, bottom_right, bottom, top_left, bottom_left };
          _numNeighbors[i] = 6;
        }
      }
    }

  protected:

    Real age_ = {};

  public:

  Natural _width_total = {}; // The maximum width in cells of the PSM
  Natural _cells_total = {}; // The total number of cells of the PSM (total width * total height)
  CUDA_Array<int, 6>* _neighbors;
  Parameter_Set _parameter_set = {};
  //Real* factors_perturb;
  //Real** factors_gradient;
  cell_param<NUM_REACTIONS + NUM_DELAY_REACTIONS + NUM_CRITICAL_SPECIES> _cellParams;
  Real max_delays[NUM_SPECIES] = {};  // The maximum number of time steps that each specie might be accessed in the past
  Natural* _numNeighbors;

  // Sizes

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

}

template <typename T>
Real dense::Context<T>::getCritVal(critspecie_id rcritsp) const {
  return owner_->_cellParams[NUM_REACTIONS+NUM_DELAY_REACTIONS+rcritsp][cell_];
}

template <typename T>
Real dense::Context<T>::getRate(reaction_id reaction) const {
    return owner_->_cellParams[reaction][cell_];
}

template <typename T>
Real dense::Context<T>::getDelay(delay_reaction_id delay_reaction) const {
    return owner_->_cellParams[NUM_REACTIONS+delay_reaction][cell_];
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


template<int N, class T>
IF_CUDA(__host__ __device__)
dense::cell_param<N, T>::cell_param(Natural width_total, Natural cells_total)
: cell_count_{cells_total},
  simulation_width_{width_total},
  _array{new T[_height * cell_count_]} {}

#endif
