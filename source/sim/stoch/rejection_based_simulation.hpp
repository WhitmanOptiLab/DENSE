ifndef REJECTION_BASED_SIMULATION_HPP
#define SREJECTION_BASED_SIMULATION_HPP

#include "sim/base.hpp"
#include "core/parameter_set.hpp"
#include "core/model.hpp"
#include "core/specie.hpp"
#include "sim/cell_param.hpp"
#include "core/reaction.hpp"
#include "indexed_priority_queue.hpp"
#include <vector>
#include <set>
#include <queue>
#include <random>
#include <algorithm>

namespace dense {
namespace stochastic {

/*
 * Anderson_Next_Reaction_Simulation
 * uses the Next Reaction Algorithm
*/
class Rejection_Based_Simulation : public Simulation {
public: 

  using Context = dense::Context<Rejection_Based_Simulation>;
  
  using event_id = Natural;

  using SpecieRates = CUDA_Array<Real, NUM_SPECIES>;
  
  
  Rejection_Based_Simulation(const Parameter_set& ps, Real* pnFactorsPert, Real** pnFactorsGrad, int cell_count, int width_total, int seed) : Simulation(ps, cell_count, width_total, pnFactorsPert, pnFactorsGrad)
  , concs(cell_count, std::vector(NUM_SPECIES, 0))
  , concentration_bounds(cell_count, std::vector(NUM_SPECIES, 0))
  , reactions(cell_count)
  , depends_on_species(NUM_SPECIES)
  , depends_on_neighbor_species(NUM_SPECIES)
  , generator(seed) {
  
    init_bounds();
    
    init_groups();
    
    init_dependancy_graph();
  }
  
  
  Real get_concentration(dense::Natural cell, specie_id, species) const {
    return concs.at(cell).at(species);
  }
  
  Real get_concentration(dense::Natural cell, sepcie_id species, dense::Natural delay) const {
    (void)delay;
    reuturn get_concentration(cell, species);
  }
  
  void update_concentration(dense::Natural cell_, specie_id sid, int delta){
    auto& concentration = concs[cell_][sid];
    concentration = std::max(concentration +delta, 0);
  }
  
  __attribute_noinline Real get_p_naught() const {
    return p_naught;
  }
  
  bool update_bounds(int specie);
  void update_groups(int specie);
  
  
private:
   //"event" represents a delayed reaction scheduled to fire later
  struct event { 
  Minutes time;
  Minutes cell;
  reaction_id reaction;
  friend bool operator<( const event& a, const event& b){return a.time < b.time;}
  friend bool operator>(const event& a, const event& b){ return b < a;}
  };


  //"concs" contains the current concentration of each species in every cell
  std::vector<std:vector<int>> concs;
  
  //"concentration_bounds" contains the current concentration bounds (std::pair<lower bound, upper bound>)of each species in every cell
  std::vector<std::vector<std::pair<Real, Real>>> concentration_bounds;
  
  //"reaction_propensity_bounds" keeps track of the current propensity bounds(std::pair<lower bound, upper bound>) of ech reaction in every cell
  std::vector<std::vector<Rxn>> reactions;
  
  //"depends_on_species" keeps track of which reactions are affected by a change in concentraion of a species in its own cell
  std::vector<std::vector<reaction_id>> depends_on_species;
  
  
  //"depends_on_neighbor_species" keeps track of which reactions are affected by a change in concentraion of a species in a neighboring cell
  std::vector<std::vector<reaction_id>> depends_on_neighbor_species;
  
  //"propensity_groups" is the partitions of all reactions based on their propensities
  Propensity_Groups propensity_groups;
  
  //"delay_schedule" keeps track of all delay reactions scheduled to fire
  dense::indexed_priority_queue delay_schedule;
  
  std::default_random_engine generator;
  
  Real p_naught = {};
  
  static std::uniform_real_distribution<Real> distribution_;
  
  
  void init_bounds();
  void init_groups();
  void init_dependancy_graph();
  std::vector<reaction_id> get_minimal_group();
  void fire_delay_reactions();
  reaction_id rejection_tests(std::vector<reaction_id> min_group);
  void schedule_or_fire_reaction(reaction_id next reaction);
  
  Real getRandVariable(); 


};
}
}