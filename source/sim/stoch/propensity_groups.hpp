#ifndef PROPENSITY_GROUPS
#define PROPENSITY_GROUPS

#include <cmath>
#include "rxn_struct.hpp"
#include "rejection_based_simulation.hpp"
#include "sim/cell_param.hpp"
#include "model_impl.hpp"
#include "core/model.hpp"
#include <limits>
#include <iostream>
#include <cmath>
#include <set>
#include "random_selector.hpp"

namespace dense {
namespace stochastic {
  
  class Propensity_Groups{
  public:
    
    Propensity_Groups() : chooser(1) {
    };
    
    void init_propensity_groups(std::vector<Rxn>& reactions){
      bool temp = false;
      for(Rxn reaction : reactions){
        place_in_group(reaction, true, temp);
      }
      init_p_values();
      init_organized_groups();
      chooser = nonuniform_int_distribution<int>(organized_p_values);
    }
  
    
    void update_groups( std::vector<Rxn>& old_reactions, std::vector<Rxn>& new_reactions){
      
      for(size_t i = 0; i < old_reactions.size(); i++){
        
        int current_group = group_index(old_reactions[i].get_index());
        int reaction_index = find_reaction_index(old_reactions[i]);
        update_p_value(old_reactions[i], false);
        if(reaction_index >= 0){
          if(groups[current_group].size() <= 1){
            groups[current_group].clear();
          }else{
            groups[current_group].erase(groups[current_group].begin()+reaction_index);
          }
          /*  if(groups[current_group].empty()){
          *  std::cout << "deleting group \n";
          * groups.erase(groups.begin() + current_group);
          *    std::cout << "deleting map element \n" <<group_map.size() << ' ' << current_group << '\n';
          *    group_map.erase(group_map.begin() + current_group);
          *    std::cout << "deleting p value \n";
          *    p_values.erase(p_values.begin()+current_group);
          *  }
          */
        }
        else{
          std::cout << "Error: invalid reaction has index: " << reaction_index << '\n'
          << "group has size: " << groups[current_group].size() << '\n';
        }
        
      bool new_group = false;
      place_in_group(new_reactions[i], false, new_group);
      int index = new_reactions[i].get_index();
      if(new_group){
        organized_groups.push_back(index);
        organized_p_values.push_back(p_values[group_index(index)]);
        reorganize();
      }
      update_p_value(new_reactions[i], true);
    }
    chooser = nonuniform_int_distribution<int>(organized_p_values);
    }
    
  
    
  int get_minimal_group_l(std::default_random_engine& gen){  

    return organized_groups[chooser(gen)];
    }
    
  
  std::vector<Rxn> get_group_at_index(int l){ return groups[group_index(l)];}
  
    
    
  Real get_p_naught(){ return p_naught;}
    
  private:
    
    std::vector<int> group_map;
    std::vector<std::vector<Rxn>> groups; 
    std::vector<Real> p_values;
    std::vector<int> organized_groups;
    std::vector<Real> organized_p_values; 
    Real p_naught;
    nonuniform_int_distribution<int> chooser;
    
    
    
    void place_in_group(Rxn reaction, bool initializing, bool& made_new_group) {
      
     const int p = reaction.get_index();
     const int current_group = group_index(p);
      
    if(initializing){
      (void)made_new_group;
    }  
      
      //Create new group
      if(current_group == -1){
        if(!initializing){
          made_new_group = true;
        }
        std::vector<Rxn> to_insert;
        to_insert.push_back(reaction);
        groups.push_back(to_insert);
        p_values.push_back(0.0);
        group_map.push_back(p);
        sort_groups();
        }
      
      //add to existing group
      else {
        groups[current_group].push_back(reaction);
        if(!initializing){
          made_new_group = false;
        }
      }
    }
  
    
    
    void init_p_values(){
      p_naught = 0;
      for(size_t i = 0; i < groups.size(); i++){
        Real p_i = 0;
        for(size_t j = 0; j <groups[i].size(); j++){
          p_i += groups[i][j].upper_bound;
        }
        p_naught += p_i;
        p_values[i] = p_i;
      }  
    }
    
    void init_organized_groups(){
      for(size_t i =0; i < p_values.size(); i++){
          organized_groups.push_back(group_map[i]);
          organized_p_values.push_back(p_values[i]);
      }
      reorganize();
    }
    
    
    void update_organized_p_value(int index, Real new_p){
      for(size_t i =0; i < organized_groups.size(); i++){
        if(organized_groups[i] == index){
          organized_p_values[i] = new_p;
        }
      }
      reorganize();  
    }
    
    void reorganize(){
      size_t i = 1;

      while(i <organized_p_values.size()){
        Real p_val = organized_p_values[i];
        int group = organized_groups[i]; 
        int j = i - 1;
        while((j >= 0) && (organized_p_values[j] > p_val)){
          organized_p_values[j+1] = organized_p_values[j];
          organized_groups[j+1] = organized_groups[j];
          j = j - 1;
        }
        organized_p_values[j+1] = p_val;
        organized_groups[j+1] = group;
        i++;
      }
    }
    
    void sort_groups(){
      size_t i = 1;
      while(i < groups.size()){
        int group_val = group_map[i];
        Real p_val = p_values[i];
        std::vector<Rxn> group = groups[i];
        int j = i - 1;
        while((j >= 0) && (group_map[j] > group_val)){
          group_map[j+1] = group_map[j];
          p_values[j+1] = p_values[j];
          groups[j+1] = groups[j];
          j = j-1;
        }
        group_map[j+1] = group_val;
        p_values[j+1] = p_val;
        groups[j+1] = group;
        i++;
      }
    }
    void update_p_value(Rxn reaction, bool adding_to){
      int current_group = group_index(reaction.get_index());
      if(adding_to){
        p_values[current_group] += reaction.upper_bound;
        p_naught += reaction.upper_bound;
        update_organized_p_value(reaction.get_index(), p_values[current_group]);
      }
      else{
        p_values[current_group] -= reaction.upper_bound;
        p_naught -= reaction.upper_bound;
        update_organized_p_value(reaction.get_index(), p_values[current_group]);
      }
    }
    
    
    
    int group_index(int index){
      int s = group_map.size()-1;
      int i = 0;
      while(i <= s){
        int m = (s+i)/2;
        if(group_map[m] == index){
          return m;
        }
        if(group_map[m] < index){
          i = m+1;
        }
        else if(group_map[m] > index){
          s = m-1;
        }
      }
      return -1;
    }
    int find_reaction_index(Rxn reaction){
      int current_group = group_index(reaction.get_index());
      for(size_t i = 0; i < groups[current_group].size(); i++){
        if(groups[current_group][i] == reaction){
          return i; 
        }
      }
      std::cout << "reaction with bound " << reaction.upper_bound << " is not in group with index " << group_map[current_group] << '\n';
      return -1;
      
    }
  };
  
  
  
}
} 
#endif
