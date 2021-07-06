#ifndef PROPENSITY_GROUPS
#define PROPENSITY_GROUPS

#include <cmath>
#include "rxn_struct.hpp"
#include "rejection_based_simulation.hpp"
#include "sim/cell_param.hpp"
#include "model_impl.hpp"
#include "core/model.hpp"
#include <algorithm>
#include <limits>
#include <iostream>
#include <cmath>
#include <set>
#include "datastructures/random_selector.hpp"

namespace dense {
namespace stochastic {
  
  class Propensity_Groups{
  public:
    
    Propensity_Groups() : chooser(1) {
    };
    
    void init_propensity_groups(std::vector<Rxn>& reactions) {
      for(Rxn reaction : reactions){
        (void) add_reaction(reaction);
      }
      chooser = nonuniform_int_distribution<int>(p_values);
    }
  
    
    void update_groups( std::vector<Rxn>& old_reactions, std::vector<Rxn>& new_reactions){
      
      for(size_t i = 0; i < old_reactions.size(); i++){
        remove_reaction(old_reactions[i]);
        add_reaction(new_reactions[i]);
      }
      chooser = nonuniform_int_distribution<int>(p_values);
    }
    
    int select_group_by_pl(std::default_random_engine& gen){  
      return chooser(gen);
    }
    
  
    std::vector<Rxn> get_group_at_index(int i){ return groups[i];}
    std::vector<Rxn> get_group_at_rank(int l){ return groups[group_map.at(l)];}
  
    
    
    Real get_p_naught(){ return p_naught;}
    
  private:
    
    std::map<int, int> group_map; // map of group rank to group index
    std::vector<std::vector<Rxn>> groups; //stores vector of reactions
    std::vector<Real> p_values; // stores value of upper bounds for each group of reactions
    /*std::vector<int> organized_groups;
    std::vector<Real> organized_p_values;*/ 
    Real p_naught; //stores sum of upper bounds
    nonuniform_int_distribution<int> chooser;
    
    
    
    bool add_reaction(Rxn reaction) {
      const int group_rank = reaction.get_group_rank();
      bool made_new_group = false;
      
      //Create new group
      if(group_map.find(group_rank) == group_map.end()){
        made_new_group = true;
        std::vector<Rxn> to_insert;
        to_insert.push_back(reaction);
        group_map[group_rank] = groups.size();
        groups.push_back(to_insert);
        p_values.push_back(0.0);
      }
      //add to existing group
      else {
        groups[group_map.at(group_rank)].push_back(reaction);
      }

      p_values[group_map.at(group_rank)] += reaction.upper_bound;
      p_naught += reaction.upper_bound;
      return made_new_group;
    }
    
    void remove_reaction(Rxn reaction) {
      int gid = group_map.at(reaction.get_group_rank());
      auto reaction_in_group = find(groups[gid].begin(), groups[gid].end(), reaction);
      groups[gid].erase(reaction_in_group);
      p_values[gid] -= reaction.upper_bound;
      p_naught -= reaction.upper_bound;
    }
  };
  
  
  
}
} 
#endif
