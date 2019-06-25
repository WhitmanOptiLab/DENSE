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
  
  class Propensity_Groups{
  public:
    
    Propensity_Groups(std::vector<std::vector<std::pair<Real, Real>>> bounds, Natural ctotal, Natural rtotal) {
      
      for(Natural c = 0; c < ctotal; ++c){
        for(Natural r = 0; r < rtotal; ++r){
          Rxn rxn;
          rxn.upper_bound = bounds[c][r].second();
          rxn.cell = c;
          rxn.reaction = r;
          
          place_in_group(rxn);
        }
      }
      init_p_values();
    }
      
    void update_groups(std::pair<Natural, Natural> index, Natural current_group, std::pair<Real, Real> bounds){

      int i = 0;
      Rxn rxn;
      rxn.cell = index.first();
      rxn.reaction = index.second();
      rxn.upper_bound = bounds.second();
      int group_index = group_at(current_group);

      if(group_index == -1){
        std::cout << "In update_groups: invalid group index" << "\n";
      }

      while(i < groups[group_index].size()){

        if(groups[group_index][i] == rxn){

          groups[group_index].erase(groups[group_index].begin()+ i);
          update_p_value(group_index);
          break;

         }

        i++;
      }

        place_in_group(rxn);
      
        update_p_value(group_at(rxn.get_index));
   }
    
  
    std::vector<Rxn> get_minimal_group(Real r_1){
      Real test_factor = p_naught * r_1;
      Real sum_p_values = 0;
      int i = 0;
      
      while(i < p_values.size()){
        sum_p_values += p_values[i];
        if(sum_p_values > test_factor){
          return(groups[group_at(i)]);
        }
        i++;
      }
    }
    
    Real get_p_naught(){ return p_naught;}
    
  private:
    
    std::vector<int> current_indexes;
    std::vector<std::vector<Rxn>> groups; 
    std::vector<Real> p_values;
    Real p_naught;
    
    
    void place_in_group(Rxn reaction) {
      
      int p = reaction.get_index();
      
      //Create new group
      if(group_at(p) == -1){
        
        int i = 0;
        
        while(i < (current_indexes.size() -1)){
          
          if(current_indexes[i] < p && current_indexes[i+1] > p){
            current_indexes.insert(i+1, p);
            std::vector<Rxn> to_insert(reaction);
            groups.insert(i+1, to_insert);
            return;
          }
          i++;
        }
      }
      
      //add to existing group
      else {
        groups[group_at(p)].push_back(reaction);
        return;   
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
      int current_group = group_at(index);
      p_naught -= p_values[current_group];
      int new_p = 0;
      for(int i = 0; i < groups[current_group].size(); i++){
        new_p += groups[current_group][i].upper_bound;
      }
      p_values[current_group] = new_p;
      p_naught += new_p;
    }
    
    
    int group_at(int index){
      
      int group_index = -1;
      int i = 0;
      
      while(i < current_indexes.size()){
        if(current_indexes[i] == index){
          group_index = i;
          break;
        }
        i++;
      }
      
      return group_index;
    }
    
    
  };
  
struct Rxn {
  Natural cell;
  Natural reaciton;
  Real upper_bound;
  friend bool opperator==(const Rxn& a, const Rxn& b){ return (a.cell == b.cell && a.reaction == b.reaction);}
  int get_index() {return (int)(std::log2(upper_bound));}
};
}