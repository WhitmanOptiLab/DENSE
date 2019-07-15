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

namespace dense {
namespace stochastic {
  
  class Propensity_Groups{
  public:
    
    Propensity_Groups() = default;
    
    void init_propensity_groups(std::vector<Rxn>& reactions){
      for(Rxn reaction : reactions){
        place_in_group(reaction);
      }
      init_p_values();
    }
  
    
    void update_groups( std::vector<Rxn>& old_reactions, std::vector<Rxn>& new_reactions){
      
      for(size_t i = 0; i < old_reactions.size(); i++){
        update_p_value(old_reactions[i], false);
        int current_group = group_index(old_reactions[i].get_index());
        int reaction_index = find_reaction_index(old_reactions[i]);
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
      place_in_group(new_reactions[i]);
      update_p_value(new_reactions[i], true);
    }
    }
    
  
    
  int get_minimal_group_index(Real r_1){
      Real test_factor = p_naught * r_1;
      Real sum_p_values = 0;
      size_t i = 0;
      while(i < p_values.size()){
        sum_p_values += p_values[i];
        if(sum_p_values > test_factor){
          return i;
        }
        i++;
      }
      std::cout << "sum of p values is: " << sum_p_values << '\n' <<
        "text factor is: " << test_factor << '\n' << "this is not ok \n";
      return -1;
    }
    
  
  std::vector<Rxn> get_group_at_index(int l){ return groups[l];}
    
  int get_l_value(int index){
    return group_map[index];
  }
  
    
    
  Real get_p_naught(){ return p_naught;}
    
  private:
    
    std::vector<int> group_map;
    std::vector<std::vector<Rxn>> groups; 
    std::vector<Real> p_values;
    Real p_naught;
    
    
    
    void place_in_group(Rxn reaction) {
      
     const int p = reaction.get_index();


     const int current_group = group_index(p);
      
      //Create new group
      if(current_group == -1){
        std::vector<Rxn> to_insert;
        to_insert.push_back(reaction);
        
        if(group_map.size() == 0){
          group_map.push_back(p);
          groups.push_back(to_insert);
          p_values.push_back(0.0);
         
          }
          else if(group_map.size() == 1){
              if(group_map[0] > p){
                group_map.insert(group_map.begin(), p);
                groups.insert(groups.begin(), to_insert);
                p_values.insert(p_values.begin(), 0.0);
              }
              else{
           
                group_map.push_back(p);
                groups.push_back(to_insert);
                p_values.push_back(0.0);
              }
          }
            
            else{
              if( group_map[0] > p){
                  group_map.insert(group_map.begin(), p);
                  groups.insert(groups.begin(), to_insert);
                  p_values.insert(p_values.begin(),0.0);
                  return;
                }
              else{
                for(size_t i = 0; i < group_map.size(); i++){
                  if(i+1 < group_map.size()){
                    if((group_map[i] < p) && (group_map[i+1] > p)){
                        group_map.insert((group_map.begin()+i+1), p);
                        groups.insert((groups.begin()+i+1), to_insert);
                        p_values.insert((p_values.begin()+i+1),0.0);
                        return;
                    }
                  } else{
                        group_map.push_back(p);
                        groups.push_back(to_insert);
                        p_values.push_back(0.0);
                        return;
                  }
                }
              }
            }
        }
      
      //add to existing group
      else {
        groups[current_group].push_back(reaction);
      }
    }
  
    
    
    void init_p_values(){
      p_naught = 0;
      for(size_t i = 0; i < groups.size(); i++){
        int p_i = 0;
        for(size_t j = 0; j <groups[i].size(); j++){
          p_i += groups[i][j].upper_bound;
        }
        p_naught += p_i;
        p_values[i] = p_i;
      }  
    }
    
    
    
    void update_p_value(Rxn reaction, bool adding_to){
      int current_group = group_index(reaction.get_index());
      if(adding_to){
        p_values[current_group] += reaction.upper_bound;
        p_naught += reaction.upper_bound;
      }
      else{
        p_values[current_group] -= reaction.upper_bound;
        p_naught -= reaction.upper_bound;
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