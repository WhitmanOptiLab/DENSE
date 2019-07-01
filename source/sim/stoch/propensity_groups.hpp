#include <cmath>
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
    
    
    
    Propensity_Groups(std::vector<Rxn> reactions){
      
      for(Rxn reaction : reactions){
        place_in_group(reaction);
      }
      init_p_values();
    }
  
    
    
    void update_groups(std::vector<Rxn> old_reactions, std::vector<Rxn> new_reactions){
      
      //Remove old reactions from their groups
      for(auto reaction : old_reactions){
        int i = 0;
        int index = raction.get_index();
        
        if(index == -1){
          std::cout << "In update_groups: invalid group index" << "\n";
        }

        while(i < groups[index].size()){

          if(groups[index][i] == rxn){

            groups[index].erase(groups[index].begin()+ i);
            update_p_value(index);
            break;
           }

          i++;
        }
      }
      
      
      //Add in new reactions
      for(Rxn rxn : new_reactions){
        place_in_group(rxn);
        update_p_value(group_index(rxn.get_index));
      }
    }
      
    
  
    
  int get_minimal_group_index(Real r_1){
      Real test_factor = p_naught * r_1;
      Real sum_p_values = 0;
      int i = 0;
      
      while(i < p_values.size()){
        sum_p_values += p_values[i];
        if(sum_p_values > test_factor){
          return group_map[i];
        }
        i++;
      }
    }
    
  
  std::vector<Rxn> get_group_at_index(int l){ return groups[group_index(l)];}
  
    
    
  Real get_p_naught(){ return p_naught;}
    
  private:
    
    std::vector<int> group_map;
    std::vector<std::vector<Rxn>> groups; 
    std::vector<Real> p_values;
    Real p_naught;
    
    
    
    void place_in_group(Rxn reaction) {
      
      int p = reaction.get_index();
      int current_group = group_index(p);
      
      //Create new group
      if(current_group == -1){
        
        int i = 0;
        
        std::vector::iterator map_it = group_map.begin();
        std::vector::iterator groups_it = groups.begin();
        
        while(i < (group_map.size() -1)){
          
          if(group_map[i] < p && group_map[i+1] > p){
            group_map.insert((map_it+i+1), p);
            std::vector<Rxn> to_insert(reaction);
            groups.insert((groups_it+i+1), to_insert);
            return;
          }
          i++;
        }
      }
      
      //add to existing group
      else {
        int current_group_size = groups[current_group].size();
        std::vector::iterator it = groups.begin();
        
        for(int i =0; i < current_group_size; i++){
          if(i == current_group_size-1){
            groups[current_group].push_back(reaction);
          } else if(groups[current_group][i] < reaction && reaction < groups[current_group][i+1]){
              groups[current_group].insert((it+i+1), reaction);
          }
        }
      }
    }
    
    
    
    void init_p_values(){
      p_naught = 0;
      for(int i = 0; i < groups.size(); i++){
        int p_i = 0;
        for(int j = 0; j <groups[i].size(); j++){
          p_i += groups[i][j].upper_bound;
        }
        p_naught += p_i;
        p_values.push_back(p_i);    
      }  
    }
    
    
    
    void update_p_value(int index){
      int current_group = group_index(index);
      p_naught -= p_values[current_group];
      int new_p = 0;
      for(int i = 0; i < groups[current_group].size(); i++){
        new_p += groups[current_group][i].upper_bound;
      }
      p_values[current_group] = new_p;
      p_naught += new_p;
    }
    
    
    
    int group_index(int index){
      int s = group_map.size();
      int i = 0;
      int m;
      
      while(i <= s){
        m = i + ((r - i)/2);
        
        if(group_map[m] == index){
          return m;
        }
        else if(group_map[m] < index){
          i = m + 1;
        } else { 
          i = m - 1; 
        }
      }
      
      return -1;
    }
  };
  
  
  
struct Rxn {
  Natural cell;
  Natural reaciton;
  Real upper_bound;
  Real lower_bound;
  friend bool opperator==(const Rxn& a, const Rxn& b){ return (a.cell == b.cell && a.reaction == b.reaction);}
  friend bool opperator<(const Rxn& a, const Rxn& b){return (a.upper_bound < b.upper_bound);}
  friend bool opperator> (const Rxn& a, const Rxn& b){return (b < a);}
  
  int get_index() {return (int)(std::log2(upper_bound));}
  
};
  
}
}