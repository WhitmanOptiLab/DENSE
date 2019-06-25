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




rejection
}
}