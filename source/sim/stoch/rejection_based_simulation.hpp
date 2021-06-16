#ifndef REJECTION_BASED_SIMULATION_HPP
#define REJECTION_BASED_SIMULATION_HPP

#include "sim/base.hpp"
#include "core/parameter_set.hpp"
#include "core/model.hpp"
#include "core/specie.hpp"
#include "sim/cell_param.hpp"
#include "core/reaction.hpp"
#include "propensity_groups.hpp"
#include <vector>
#include <set>
#include <queue>
#include <random>
#include <algorithm>
#include <chrono>

namespace dense {
namespace stochastic {

/*
 * Anderson_Next_Reaction_Simulation
 * uses the Next Reaction Algorithm
*/
class Rejection_Based_Simulation : public Simulation {
public: 
  
  using Context = dense::Context<Rejection_Based_Simulation>;
  
  using event_id = dense::Natural;

  using SpecieRates = CUDA_Array<Real, NUM_SPECIES>;
  
private:
   //"concs" contains the current concentration of each species in every cell
  std::vector<std::vector<int>> concs;
  
  //"concentration_bounds" contains the current concentration bounds (std::pair<lower bound, upper bound>)of each species in every cell. Array row 0 is lower bounds, array row 1 is upper bounds. 
  std::vector<std::vector<Real>> concentration_bounds[2];
  
  //"reaction_propensity_bounds" keeps track of the current propensity bounds(std::pair<lower bound, upper bound>) of ech reaction in every cell
  std::vector<Rxn> reactions;
  
  //"depends_on_species" keeps track of which reactions are affected by a change in concentraion of a species in its own cell
  std::vector<reaction_id> depends_on_species[NUM_SPECIES];
  
  
  //"depends_on_neighbor_species" keeps track of which reactions are affected by a change in concentraion of a species in a neighboring cell
  std::vector<reaction_id> depends_on_neighbor_species[NUM_SPECIES];
  
  //"depends_on_reaction" is a dependency graph that shows which specie concentrations are affected by a given reaction firing
  
  std::vector<specie_id> depends_on_reaction[NUM_REACTIONS];
  
  //"propensity_groups" is the partitions of all reactions based on their propensities
  Propensity_Groups propensity_groups;
  
  //"delay_schedule" keeps track of all delay reactions scheduled to fire
  std::priority_queue<Delay_Rxn, std::vector<Delay_Rxn>,std::greater<Delay_Rxn>> delay_schedule;
  
  std::default_random_engine generator;
  
  double delta;
  
  int y;
  
  
  
  static std::uniform_real_distribution<Real> distribution_;
  
   void init_bounds();
  //
  void init_dependancy_graph();
  //
  bool fire_delay_reactions(Minutes tau);
  //
  bool rejection_tests( Rxn& rxn,int min_group_index);
  //
  void schedule_or_fire_reaction(Rxn& next_reaction);
  //
  void fire_reaction(Rxn& rxn);
  //
  bool check_bounds(std::vector<std::pair<dense::Natural, dense::Natural>>& changed_species, Rxn fired_reaction);
  //
  void update_bounds(std::vector<std::pair<dense::Natural, dense::Natural>>& to_update);
  //
  Real get_real_propensity(Rxn rxn);
  //
  Real getRandVariable(); 
public: 

  
  
 
   Rejection_Based_Simulation(const Parameter_Set& ps, Real* pnFactorsPert, Real** pnFactorsGrad, int seed, std::vector<int> conc, NGraph::Graph adj_graph) :  Simulation(ps, std::move(adj_graph), pnFactorsPert, pnFactorsGrad)
    , concs(cell_count(), conc)
    , delay_schedule()
    , generator(seed){
    init_bounds();
    propensity_groups.init_propensity_groups(reactions);
    init_dependancy_graph();
  }
  
    Rejection_Based_Simulation(const Parameter_Set& ps, Real* pnFactorsPert, Real** pnFactorsGrad, int seed, std::vector<int> conc, NGraph::Graph adj_graph, double d, int yval) :  Simulation(ps, std::move(adj_graph), pnFactorsPert, pnFactorsGrad)
    , concs(cell_count(), conc)
    , delay_schedule()
    , generator(seed)
    , delta(d)
    , y(yval){
    init_bounds();
    propensity_groups.init_propensity_groups(reactions);
    init_dependancy_graph();
  }
  
  Real get_concentration(dense::Natural cell, specie_id species) const {
    return concs.at(cell).at(species);
  }
  
  Real get_concentration(dense::Natural cell, specie_id species, dense::Natural delay) const {
    (void)delay;
    return get_concentration(cell, species);
  }
  
  void update_concentration(dense::Natural cell_, specie_id sid, int delta){
    auto& concentration = concs[cell_][sid];
    concentration = std::max(concentration +delta, 0);
  }
  

  Minutes age_by(Minutes duration);
  
  Real calculate_neighbor_average (dense::Natural cell, specie_id species, dense::Natural delay) const {
    (void)delay;
    Real sum = 0;
    for (dense::Natural i = 0; i < neighbor_count_by_cell_[cell]; ++i) {
      sum += concs[neighbors_by_cell_[cell][i]][species];
    }
    return sum / neighbor_count_by_cell_[cell];
  }
  
private:
   //"event" represents a delayed reaction scheduled to fire later
  struct event { 
  Minutes time;
  Minutes cell;
  reaction_id reaction;
  friend bool operator<( const event& a, const event& b){return a.time < b.time;}
  friend bool operator>(const event& a, const event& b){ return b < a;}
  };


 
  


};

  

class ConcentrationContext {
      public:
        ConcentrationContext(std::vector<Real> concentrations, Rejection_Based_Simulation& sim, Natural cell) :
            concs(concentrations), ctxt(sim, cell) {};
        Real getCon(specie_id specie, int = 0) const {
          return concs.at(specie);
        }
        Real getCon(specie_id specie){
          return concs.at(specie);
        }
        Real getRate(reaction_id reaction) const { return ctxt.getRate(reaction);};
        Real getDelay(delay_reaction_id reaction) const { return ctxt.getDelay(reaction); }
        Real getCritVal(critspecie_id reaction) const { return ctxt.getCritVal(reaction); }
        Real calculateNeighborAvg(specie_id sp, int = 0) const {
           (void)sp;
            return 0.0;
        }
      private:
        
        std::vector<Real> concs;
        Context<Rejection_Based_Simulation> ctxt;
    };

}
using stochastic::Rejection_Based_Simulation;
}
#endif
