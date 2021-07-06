#include <cmath>
#include "rejection_based_simulation.hpp"
#include "rxn_struct.hpp"
#include "sim/cell_param.hpp"
#include "model_impl.hpp"
#include "core/model.hpp"
#include "propensity_groups.hpp"
#include <limits>
#include <iostream>
#include <cmath>
#include <set>

namespace dense {
namespace stochastic {

/*
 * SIMULATE
 * main simulation loop
 * notifies observers
 * precondition: t=0
 * postcondition: ti>=time_total
*/

std::uniform_real_distribution<Real> Rejection_Based_Simulation::distribution_ = std::uniform_real_distribution<Real>{0.0, 1.0};

CUDA_AGNOSTIC
Minutes Rejection_Based_Simulation::age_by(Minutes duration){
  auto end_time = age() + duration;
  auto start = std::chrono::high_resolution_clock::now();
  Simulation::step(true);
  while(age() < end_time){
    std::vector<std::pair<dense::Natural,dense::Natural>> changed_species;
    bool all_delays_fired = false; 
    bool reaction_fired = false; 
    Rxn reaction_to_be_fired;
    while(!reaction_fired){
    auto group_index = propensity_groups.select_group_by_pl(generator);
    auto r_2 = getRandVariable();
    Minutes tau = Minutes{(-1/propensity_groups.get_p_naught())*(std::log(r_2))};
    all_delays_fired = fire_delay_reactions(tau);
    if(all_delays_fired){
      reaction_fired = rejection_tests(reaction_to_be_fired, group_index);
      if(reaction_fired){
        schedule_or_fire_reaction(reaction_to_be_fired);
      }
      Simulation::age_by(tau);
    }
    else{
      reaction_fired = true;
      }
    }
    if(all_delays_fired){
      if(check_bounds(changed_species, reaction_to_be_fired)){
        update_bounds(changed_species);
      }
    }
  }
  auto finish = std::chrono::high_resolution_clock::now();
  Simulation::get_performance(finish - start);
  return age();
}
  
  
  
void Rejection_Based_Simulation::schedule_or_fire_reaction(Rxn& rxn){
  Delay_Rxn delay_reaction;
  delay_reaction.rxn = rxn;
  delay_reaction.delay_reaction = dense::model::getDelayReactionId(rxn.reaction);
  if(delay_reaction.delay_reaction != NUM_DELAY_REACTIONS){
    delay_reaction.delay =( age() + Minutes{Context(*this, delay_reaction.rxn.cell).getDelay(delay_reaction.delay_reaction)});
    delay_schedule.push(delay_reaction);
  }
  else{ 
    fire_reaction(rxn);
  }
  
}
  
void Rejection_Based_Simulation::fire_reaction(Rxn& rxn){
  const reaction_base& rn = dense::model::getReaction(rxn.reaction);
  const specie_id* specie_deltas = rn.getSpecieDeltas();
  Simulation::step(false);
  for(int i = 0; i < rn.getNumDeltas(); i++){
    update_concentration(rxn.cell, specie_deltas[i], rn.getDeltas()[i]);
  }
}

void Rejection_Based_Simulation::init_bounds() {
  for(int i = 0; i <cell_count(); i++){
    std::vector<Real> temp_lower_bounds;
    std::vector<Real> temp_upper_bounds;
    
    for(size_t j = 0; j < NUM_SPECIES; j++){
      Real lower;
      Real upper;
      if(concs[i][j] >= (y/delta)){ 
        lower = concs[i][j]*(1-delta);
        upper = concs[i][j]*(1+delta);
      } 
      else{
        lower = concs[i][j] - y;
        upper = concs[i][j] + y;
      } 
      temp_lower_bounds.push_back(std::max(0.0,lower));
      temp_upper_bounds.push_back(std::max(0.0, upper));
    }
    concentration_bounds[0].push_back(temp_lower_bounds);
    concentration_bounds[1].push_back(temp_upper_bounds);
    ConcentrationContext lower_context(concentration_bounds[0], *this, i);
    ConcentrationContext upper_context(concentration_bounds[1], *this, i);
    
      
      Rxn rxn; 
      #define REACTION(name) \
      rxn.reaction = name; \
      rxn.cell = i; \
      rxn.lower_bound = std::max(0.0,dense::model::reaction_##name.active_rate(lower_context)); \
      rxn.upper_bound = std::max(0.0,dense::model::reaction_##name.active_rate(upper_context)); \
      reactions.push_back(rxn);
      #include "reactions_list.hpp"
      #undef REACTION  
    }
  }
  
  
  
void Rejection_Based_Simulation::init_dependancy_graph(){

  std::vector<specie_id> neighbor_dependencies[NUM_REACTIONS];
    std::vector<specie_id> dependencies[NUM_REACTIONS];

    class DependanceContext {
      public:
        DependanceContext(std::vector<specie_id>& neighbordeps_tofill,std::vector<specie_id>& deps_tofill) :
            interdeps_tofill(neighbordeps_tofill), intradeps_tofill(deps_tofill) {};
        Real getCon(specie_id sp, int = 0) const {
            intradeps_tofill.push_back(sp);
            return 0.0;
        };
        Real getCon(specie_id sp){
            intradeps_tofill.push_back(sp);
            return 0.0;
        };
        Real getRate(reaction_id) const { return 0.0; };
        Real getDelay(delay_reaction_id) const { return 0.0; };
        Real getCritVal(critspecie_id) const { return 0.0; };
        Real calculateNeighborAvg(specie_id sp, int = 0) const {
            interdeps_tofill.push_back(sp);
            return 0.0;
        };
      private:
        std::vector<specie_id>& interdeps_tofill;
        std::vector<specie_id>& intradeps_tofill;
    };

    #define REACTION(name) \
    const reaction<name>& r##name = dense::model::reaction_##name; \
    r##name.active_rate( DependanceContext (neighbor_dependencies[name],dependencies[name]));
    #include "reactions_list.hpp"
    #undef REACTION
  
  for(int i = 0; i < NUM_REACTIONS; i++){
    for( auto s : dependencies[i]){
      depends_on_species[s].push_back((reaction_id)i);
      depends_on_reaction[i].push_back(s);
    }
    for(auto s : neighbor_dependencies[i]){
      depends_on_neighbor_species[s].push_back((reaction_id)i);
    }
  }
}


  
bool Rejection_Based_Simulation::fire_delay_reactions(Minutes tau){
  if(!delay_schedule.empty()){
    Minutes delay_count = Minutes{0};
    while(delay_schedule.top().delay < age() + tau){
      auto reaction = delay_schedule.top();
      delay_schedule.pop();
      fire_reaction(reaction.rxn);
      delay_count += (reaction.delay- age());
      std::vector<std::pair<dense::Natural, dense::Natural>> changed;
      if(check_bounds(changed, reaction.rxn)){
        Simulation::age_by(delay_count);
        update_bounds(changed);
        return false;
      }
    }
  }
  return true;
}

bool Rejection_Based_Simulation::rejection_tests(Rxn& rxn, int group_index){
  auto group = propensity_groups.get_group_at_index(group_index);
  auto group_rank = group[0].get_group_rank();
  Real two_power = pow(2,group_rank);
  bool mu_found = false;
  int mu;
  while(!mu_found){
    Real r_2 = getRandVariable();
    mu = (int)(group.size() * r_2);
    
    #define DOUBLE_PRECISION_VARIABLE
    #ifdef DOUBLE_PRECISION_VARIABLE
    Real r_3 = (group.size() * r_2) -mu;
   
    #else
    
    Real r_3 = getRandVariable();
    #endif 
    #undef DOUBLE_PRECISION_VARIABLE
    if(r_3 <= (group[mu].upper_bound/(two_power))){
      mu_found = true;
    }
  }
  Real r_4 = getRandVariable();
  Rxn reaction = group.at(mu);
  bool accepted = false;
  
  assert(reaction.lower_bound <= get_real_propensity(reaction));
  assert(reaction.upper_bound >= get_real_propensity(reaction));
  
  if(r_4 <= reaction.lower_bound/reaction.upper_bound){
    
    accepted = true;
  
  }
  else {
    auto propensity = get_real_propensity(reaction);
    if(r_4 <= propensity/reaction.upper_bound){
      accepted = true;
    }
    
  }

  if(accepted){
    rxn = reaction;
  }
  else{
    (void)rxn;
  }
  
  return accepted;
}


Real Rejection_Based_Simulation::get_real_propensity(Rxn rxn){
  return std::max(dense::model::active_rate(rxn.reaction, Context(*this, rxn.cell)), Real{0}); 
}
  
bool Rejection_Based_Simulation::check_bounds(std::vector<std::pair<dense::Natural, dense::Natural>>& changed_species, Rxn fired_reaction){
  bool changed = false;
  std::vector<std::pair<dense::Natural,dense::Natural>> new_concs;
  dense::Natural c = fired_reaction.cell;
  const reaction_base& rn = dense::model::getReaction(fired_reaction.reaction);
  for(int i = 0; i < rn.getNumDeltas(); i++){
    const specie_id s = rn.getSpecieDeltas()[i];
    if(concs[c][s] < concentration_bounds[0][c][s] || concs[c][s] > concentration_bounds[1][c][s]){
      auto new_pair = std::pair<dense::Natural, dense::Natural>(c,s);
      new_concs.push_back(new_pair);
      if(!changed){
        changed = true;
      }
    }
  }
  if(changed){
    changed_species = new_concs;
  }
  return changed;
}
  
//finish
void Rejection_Based_Simulation::update_bounds(std::vector<std::pair<dense::Natural,dense::Natural>>& to_update){
  
  std::vector<Rxn> old_reactions;
  std::vector<Rxn> new_reactions;

  for(std::pair<dense::Natural,dense::Natural> specie : to_update){
    
    int current_conc = concs[specie.first][specie.second];
    Real lower;
    Real upper;
    if(current_conc >= (y/delta)){ 
      upper = current_conc*(1+delta);
      lower = current_conc*(1-delta);
   } 
    else{
      upper = current_conc + y;
      lower = current_conc - y;
      
    } 
    concentration_bounds[0][specie.first][specie.second] = std::max(0.0,lower);
    concentration_bounds[1][specie.first][specie.second] = std::max(0.0,upper);
    int begin_bounds = specie.first*NUM_REACTIONS;
      for(reaction_id r : depends_on_species[specie.second]){
            Rxn old_reaction = reactions[begin_bounds+r];
            Rxn new_reaction;
            new_reaction.cell = old_reaction.cell;
            new_reaction.reaction = old_reaction.reaction;
            ConcentrationContext lower_context(concentration_bounds[0], *this, specie.first);
            ConcentrationContext upper_context(concentration_bounds[1], *this, specie.first);
            new_reaction.lower_bound = std::max(0.0,dense::model::active_rate(new_reaction.reaction, lower_context)); 
            new_reaction.upper_bound = std::max(0.0,dense::model::active_rate(new_reaction.reaction, upper_context)); 
           if(!((old_reaction.upper_bound == new_reaction.upper_bound)&& (old_reaction.lower_bound == new_reaction.lower_bound))) {
              reactions[begin_bounds+r] = new_reaction;
              old_reactions.push_back(old_reaction);
              new_reactions.push_back(new_reaction);
            }  
      }
        for(reaction_id r : depends_on_neighbor_species[specie.second]){
          for(dense::Natural c = 0; c < neighbor_count_by_cell_[specie.first]; c++){
            Rxn old_reaction = reactions[begin_bounds+r];
            Rxn new_reaction;
            new_reaction.cell = neighbors_by_cell_[specie.first][c];
            new_reaction.reaction = old_reaction.reaction;
            ConcentrationContext lower_context(concentration_bounds[0], *this, specie.first);
            ConcentrationContext upper_context(concentration_bounds[1],*this, specie.first);
            new_reaction.lower_bound = std::max(0.0,dense::model::active_rate(new_reaction.reaction, lower_context)); 
            new_reaction.upper_bound = std::max(0.0,dense::model::active_rate(new_reaction.reaction, upper_context)); 
            int cell_reaction = (neighbors_by_cell_[old_reaction.cell][c]*NUM_REACTIONS)+r;
            if(!((old_reaction.upper_bound == new_reaction.upper_bound)&& (old_reaction.lower_bound == new_reaction.lower_bound))){
              reactions[cell_reaction] = new_reaction;
              old_reactions.push_back(old_reaction);
              new_reactions.push_back(new_reaction);
            }
        }
      }
    } 
  propensity_groups.update_groups(old_reactions, new_reactions); 
}
  
  
Real Rejection_Based_Simulation::getRandVariable(){
  return distribution_(generator);
}
  




}
}
