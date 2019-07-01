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

/*
 * SIMULATE
 * main simulation loop
 * notifies observers
 * precondition: t=0
 * postcondition: ti>=time_total
*/

std::uniform_real_distribution<Real> Rejection_Based_Simulation::distribution_ = std::uniform_real_distribution<Real>{0.0, 1.0};

void Rejection_Based_Simulation::init_bounds() {
  for(int i = 0; i < concs.size(); i++){
    for(int j = 0; j < concs[i].size(); j++){
      Real lower = concs[i][j] + (concs[i][j] * 0.2);
      Real upper = concs[i][j] + (concs[i][j] * 0.2);
      concentration_bounds[i][j] = std::make_pair<Real, Real>(lower, upper);
    }
  }
  }
  
}


}
}