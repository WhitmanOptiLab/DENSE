#include "base.hpp"
#include "sim/base.hpp"

template <typename Simulation>
void Analysis<Simulation>::when_updated_by(Simulation & simulation, std::ostream& log) {
  time = simulation.age().count();
  if (time < start_time || time >= end_time) return;
  update(simulation, log);
}
