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
    
    Propensity_Groups(std::vector<std::vector<std::pair<Real, Real>>> bounds, Natural ctotal, Natural rtotal){
      
      for(Natural c = 0; c < ctotal; ++c){
        for(Natural r = 0; r < rtotal; ++r){
          r_ubound = bounds[c][r].second();
          int pow2 (int)(std::log2(r_unbound));
          std::pair<Natural, Natural> propensity_index(c,r);
          place_in_group(pow2, propensity_index);
        }
      }
      
    void  place_in_group(int p, std::pair<Natural, Natural> propensity_index){
      int i = 0;
      while(i < (current_indexes.size() -1)){
        if(current_indexes[i] < p && current_indexes[i+1] > p){
          current_indexes.insert(i+1, propensity_index);
          
          
        }
      }
    }
    }
  
  private:
    std::vector<int> current_indexes;
    std::vector<std::vector<std::pair<Natural, Natural>> groups; 
  };
}