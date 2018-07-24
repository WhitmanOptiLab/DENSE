#ifndef SIM_BASE_HPP
#define SIM_BASE_HPP

#include "utility/common_utils.hpp"
#include "core/observable.hpp"
#include "core/parameter_set.hpp"
#include "core/model.hpp"
#include "core/specie.hpp"
#include "cell_param.hpp"
#include "core/reaction.hpp"
#include <vector>
#include <iostream>
  #include <initializer_list>
  #include <algorithm>

namespace dense {

class Simulation;

  template <typename Simulation_T = Simulation>
  class Context {

    static_assert(std::is_base_of<Simulation, Simulation_T>(),
      "Class template <typename T> dense::Context requires std::is_base_of<Simulation, T>()");
    //FIXME - want to make this private at some point
   public:

    CUDA_HOST CUDA_DEVICE
    Context(Simulation_T * owner, dense::Natural cell = 0)
    : owner_{owner}, cell_{cell} {
    }

    CUDA_HOST CUDA_DEVICE
    void advance() {
      ++cell_;
    }

    CUDA_HOST CUDA_DEVICE
    bool isValid() const;

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

    CUDA_HOST CUDA_DEVICE
    Real time() const;

  private:

    CUDA_HOST CUDA_DEVICE
    Simulation_T & owner() const;

    Simulation_T * owner_;

  public:

    Natural cell_;

  };


/* simulation contains simulation data, partially taken from input_params and partially derived from other information
 */
/* SIMULATION_BASE
 * superclass for simulation_determ and simulation_stoch
 * inherits from Observable, can be observed by Observer object
*/
class Simulation {

  public:

    using Context = dense::Context<>;

  // Sizes

  Real t = 0.0;
  Natural _width_total; // The maximum width in cells of the PSM
  //dense::Natural circumf; // The current width in cells
  //int _width_initial; // The width in cells of the PSM before anterior growth
  //int _width_current; // The width in cells of the PSM at the current time step
  //int height; // The height in cells of the PSM
  //int cells; // The number of cells in the simulation
  Natural _cells_total; // The total number of cells of the PSM (total width * total height)

  // Times and timing
  //int steps_total; // The number of time steps to simulate (total time / step size)
  //int steps_split; // The number of time steps it takes for cells to split
  //int steps_til_growth; // The number of time steps to wait before allowing cells to grow into the anterior PSM
  //bool no_growth; // Whether or not the simulation should rerun with growth

  // Neighbors and boundaries
  //array2D<int> neighbors; // An array of neighbor indices for each cell position used in 2D simulations (2-cell and 1D calculate these on the fly)
  //int active_start; // The start of the active portion of the PSM
  //int active_end; // The end of the active portion of the PSM
  CUDA_Array<int, 6>* _neighbors;

  // PSM section and section-specific times
  //int section; // Posterior or anterior (sec_post or sec_ant)
  //int time_start; // The start time (in time steps) of the current simulation
  //int time_end; // The end time (in time steps) of the current simulation

  // Mutants and condition scores
  //int num_active_mutants; // The number of mutants to simulate for each parameter set
  //double max_scores[NUM_SECTIONS]; // The maximum score possible for all mutants for each testing section
  //double max_score_all; // The maximum score possible for all mutants for all testing sections

  Parameter_Set _parameter_set;
  //Real* factors_perturb;
  //Real** factors_gradient;
  cell_param<NUM_REACTIONS + NUM_DELAY_REACTIONS + NUM_CRITICAL_SPECIES> _cellParams;
  Natural* _numNeighbors;
  //CPUGPU_TempArray<int,NUM_SPECIES> _baby_j;
  //int* _delay_size;
  //int* _time_prev;
  //double* _sets;
  //int _NEIGHBORS_2D;
  Real max_delays[NUM_SPECIES]{};  // The maximum number of time steps that each specie might be accessed in the past

  /*
   * CONSTRUCTOR
   * arg "ps": assiged to "_parameter_set", used to access user-inputted rate constants, delay times, and crit values
   * arg "cells_total": the maximum amount of cells to simulate for (initial count for non-growing tissues)
   * arg "width_total": the circumference of the tube, in cells
  */
  Simulation(Parameter_Set ps, int cells_total, int width_total, Real* factors_perturb = nullptr, Real** factors_gradient = nullptr);

  Simulation() : Simulation(Parameter_Set(), 0, 0) {}

    /*
     * CALC_NEIGHBOR_2D
     * populates the data structure "_neighbors" with cell indices of neighbors
     * follows hexagonal adjacencies for an unfilled tube
    */
    CUDA_HOST CUDA_DEVICE
    void calc_neighbor_2d() {
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

    virtual void simulate_for(Real duration) = 0;

    virtual Real get_concentration(Natural cell, specie_id species) const = 0;

    virtual Real get_concentration(Natural cell, specie_id species, Natural delay) const = 0;

    virtual Real calculate_neighbor_average(Natural cell, specie_id specie, Natural delay = 0) const = 0;

    bool was_aborted() const noexcept {
      return abort_signaled;
    }

    // Called by Observer in update
    void abort() noexcept { abort_signaled = true; }

  protected:

    void calc_max_delays(Real*, Real**);

    ~Simulation() noexcept = default;

  private:

    bool abort_signaled = false;

};

}


template <typename T>
bool dense::Context<T>::isValid() const {
  return cell_ < owner()._cells_total;
}

template <typename T>
Real dense::Context<T>::time() const {
  return owner().t;
}

template <typename T>
Real dense::Context<T>::getCritVal(critspecie_id rcritsp) const {
  return owner()._cellParams[NUM_REACTIONS+NUM_DELAY_REACTIONS+rcritsp][cell_];
}

template <typename T>
Real dense::Context<T>::getRate(reaction_id reaction) const {
    return owner()._cellParams[reaction][cell_];
}

template <typename T>
Real dense::Context<T>::getDelay(delay_reaction_id delay_reaction) const {
    return owner()._cellParams[NUM_REACTIONS+delay_reaction][cell_];
}

template <typename T>
dense::Real dense::Context<T>::getCon(specie_id sp) const {
  return owner().get_concentration(cell_, sp);
}

template <typename T>
dense::Real dense::Context<T>::getCon(specie_id sp, int delay) const {
  return owner().get_concentration(cell_, sp, delay);
}

template <typename T>
dense::Real dense::Context<T>::calculateNeighborAvg(specie_id sp, int delay) const {
  return owner().calculate_neighbor_average(cell_, sp, delay);
}

template <typename T>
T & dense::Context<T>::owner() const {
  return *(owner_ != nullptr ?
    owner_ :
    throw std::logic_error("This context does not belong to a Simulation"));
}

#endif
