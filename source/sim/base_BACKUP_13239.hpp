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
#include <cmath>
#include <iostream>
#include <initializer_list>
#include <algorithm>
#include <chrono>
#include <type_traits>
#include <utility>
#include <chrono>

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
    { simulation.get_performance()} noexcept -> std::vector<Real>;
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

    ///calculate how many reaction are fired in a second for each time step and push into performance vector
    void push_performance(std::chrono::duration<double> elapsed) noexcept;
    ///calculate how many reaction are fired in a second for each time step
    std::vector<Real> get_performance() noexcept;
    Real step(bool restart) noexcept;
    
    Real step(bool restart) noexcept;

    //add_cell_base: takes a virtual cell and adds it to the graph
    Natural add_cell_base(Natural cell){
      adjacency_graph.insert_vertex(cell);
      update_cell_count();
      Natural cell_index = find_id(cell);
      physical_cells_id_[cell_index] = cell;
      neighbor_count_by_cell_[cell_index] = 0;
      neighbors_by_cell_[cell_index] = std::vector<Natural>();
      //cell_parameters_[cell_index] initialized in the derived class      
      return cell_index;
    }
  
    //update_cell_count: will change internal variable cell count to match number of vertices
    void update_cell_count(){
      cell_count_ = adjacency_graph.num_vertices();
      cell_parameters_.cell_count_ = cell_count_;
    }
    
    //remove_cell: takes a virtual cell id and removes it from the graph
    //  -TODO: whipe out any states scheduled with the cell, for future events (delays)
    //         then exclude it from the loops calculating propensities
    Natural remove_cell_base(Natural cell){
      adjacency_graph.remove_vertex(cell);
      Natural old_index = find_id(cell); //old_cell is the index
      update_cell_count();
      for( Natural c : neighbors_by_cell_[old_index] ){
        calculate_cell_neighbors(c);
      }
      //set the virtual cell spot of the physical cell as available
      physical_cells_id_[old_index] = -1;
      //zero neighbors and cell parameters at the cells physical index
      neighbors_by_cell_[old_index] = std::vector<Natural>();
      neighbor_count_by_cell_[old_index] = 0;
      for(auto& rxn : cell_parameters_[old_index]){
        rxn = 0;
      }
      return old_index;
    }
  
    //add_edge: takes two virtual cell ids and adds an edge to the graph
    void add_edge(Natural cell1, Natural cell2){
      if(!adjacency_graph.includes_vertex(cell1)) return; 
      if(!adjacency_graph.includes_vertex(cell2)) return;
      update_cell_count();
      adjacency_graph.insert_edge(cell1, cell2);
      adjacency_graph.insert_edge(cell2, cell1);
      calculate_cell_neighbors(cell1);
      calculate_cell_neighbors(cell2);
    }
  
    //remove_edge: takes two virtual cell ids and removes the edge between them
    void remove_edge(Natural cell1, Natural cell2){
      adjacency_graph.remove_edge(cell1,cell2);
      adjacency_graph.remove_edge(cell2,cell1);
      calculate_cell_neighbors(cell1);
      calculate_cell_neighbors(cell2);
    }
  
    //find_id: adds a virtual cell into the physical_cells_id_ conversion table, the index it is stored at
    //    is the physical id and the virtual cell is the element contained 
    Natural find_id(Natural cell){
      Natural id = 0;
      Natural open = 0;
      bool found = false;
      for(auto& cell_x : physical_cells_id_){
        if( cell_x == cell){
          return id;
        }
        if( cell_x == -1 && !found){ found = true; open = id; }
        id++;
      }
      if( found ){
        physical_cells_id_[open] = cell;
        return open;
      }
      physical_cells_id_.push_back(-1);
      return find_id(cell);
    }
  
    int num_growth_cells(){
      return _num_growth_cells;
    }
    
    std::vector<int> physical_cells_id(){
      return physical_cells_id_;
    }
  
  protected:
  
    CUDA_AGNOSTIC
    Simulation () = default;

    /*
     * CONSTRUCTOR
     * arg "ps": assiged to "_parameter_set", used to access user-inputted rate constants, delay times, and crit values
     * arg "cells_total": the maximum amount of cells to simulate for (initial count for non-growing tissues)
     * arg "width_total": the circumference of the tube, in cells
    */
    Simulation(Parameter_Set parameter_set, NGraph::Graph adj_graph, Real* perturbation_factors = nullptr, Real** gradient_factors = nullptr, Natural num_growth_cell = 0) noexcept;

    CUDA_AGNOSTIC
    ~Simulation () noexcept;
  
  private:

    void calc_max_delays(Real*, Real**);
  
    /*
     * CALC_NEIGHBOR_2D
     * populates the data structure "neighbors_by_cell_" with the virtual cell indices of neighbors
     * loads graph from "adjacency_graph", either user specified or default graph from graph_constructor()
    */
    CUDA_AGNOSTIC
    void calc_neighbor_2d() noexcept {
      for ( auto p = adjacency_graph.begin(); p != adjacency_graph.end(); p++ ){
        //Update later to remove need for circumference
          Graph::vertex_set neigh = Graph::out_neighbors(p);
          circumference_ = std::max(circumference_, Natural(neigh.size()));
          auto index = adjacency_graph.node(p);
          calculate_cell_neighbors(index, false);
      }
    //Update later to remove need for circumference
      if (circumference_ == 0){
        circumference_ = 1;
      }
    }
  
      //calculate_cell_neighbors: takes a virtual cell and calculates its neighbors
    void calculate_cell_neighbors(Natural c, bool first = true){
      auto index = adjacency_graph.find(c);
      Graph::vertex_set neigh_out = Graph::out_neighbors(index);
      std::vector<Natural>* neighbors = new std::vector<Natural>;
      Natural old_index = find_id(c); //old_index is the virtual id for the physical cell c
      for ( auto cell = neigh_out.begin(); cell != neigh_out.end(); cell++ ){
        Natural neigh = *cell;
        if(first){
          neigh = find_id(*cell);
        }
        neighbors->push_back(neigh);
      }
      neighbor_count_by_cell_[old_index] = neighbors->size();
      neighbors_by_cell_[old_index] = std::move(*neighbors);
    }

    Real age_ = {};
    Natural circumference_ = {};
    Natural cell_count_ = {}; 
  protected:
    int _num_growth_cells;
    Real step_;
    //physical_cells_id: indecies 0...cell_count_ are virtual cell ids, and nodes represent the 
    //    physical cell at that virtual cell index
    //    EX: physical_cells_id_[0] = 12 means that virtual cell 0 represents the 12th physical 
    //        cell currently in the simulation
    std::vector<int> physical_cells_id_ = {};
    Parameter_Set parameter_set_ = {};
    //adjacency_graph: graph that contains physical cell nodes
    NGraph::Graph adjacency_graph;
    Real step_;
    std::vector<Real> performance_;

    
    //neighbors_by_cell_ and neighbor_count_by_cell_ are structures of virtual cell ids
    std::vector<std::vector<Natural>> neighbors_by_cell_ = {};
    Natural* neighbor_count_by_cell_ = {};

  public:
    
  
    Real max_delays[NUM_SPECIES] = {};  // The maximum number of time steps that each specie might be accessed in the past
    cell_param<NUM_PARAMS> cell_parameters_;

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
inline Real Simulation::step (bool restart) noexcept {
  if (restart){
    return Real{step_ = 0.0};
  }else{
    return Real{ step_ += 1.0 };
  }
  
}

CUDA_AGNOSTIC
inline void Simulation::push_performance(std::chrono::duration<double> elapsed) noexcept{
   performance_.push_back(step_/elapsed.count());
}

CUDA_AGNOSTIC
inline std::vector<Real> Simulation::get_performance() noexcept{
  return performance_;
}

CUDA_AGNOSTIC

CUDA_AGNOSTIC
inline Simulation::~Simulation () noexcept = default;

}

template <typename T>
Real dense::Context<T>::getCritVal(critspecie_id rcritsp) const {
  return owner_->cell_parameters_[cell_][NUM_REACTIONS+NUM_DELAY_REACTIONS+rcritsp];
}

template <typename T>
Real dense::Context<T>::getRate(reaction_id reaction) const {
  if(std::isnan(owner_->cell_parameters_[cell_][reaction])){
    throw(std::out_of_range("isnan() throw on cell_parameters_ " + std::to_string(owner_->cell_parameters_[cell_][reaction])));
  }
  return owner_->cell_parameters_[cell_][reaction];
}

template <typename T>
Real dense::Context<T>::getDelay(delay_reaction_id delay_reaction) const {
  if(std::isnan(owner_->cell_parameters_[cell_][NUM_REACTIONS+delay_reaction])){
    throw(std::out_of_range("isnan() throw on getDelay " + std::to_string(owner_->cell_parameters_[cell_][NUM_REACTIONS+delay_reaction])));
  }
  return owner_->cell_parameters_[cell_][NUM_REACTIONS+delay_reaction];
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
