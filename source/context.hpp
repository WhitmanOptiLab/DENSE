// A context defines a locale in which reactions take place and species 
//   reside
#ifndef CONTEXT_HPP
#define CONTEXT_HPP

#include "specie.hpp"
#include "concentration_level.hpp"
#include <array>


class Context {
  //FIXME - want to make this private at some point
 public:
  const int _cell;
  simulation& _simulation;
  double _avg;
  Context(simulation& sim, int cell) : _simulation(sim),_cell(cell) { }
  RATETYPE calculateNeighbourAvg(specie_id sp, int delay);
  void updateCon(const std::array<RATETYPE, NUM_SPECIES>& rates);
  const std::array<RATETYPE, NUM_SPECIES> calculateRatesOfChange();
  RATETYPE getCon(specie_id sp, int delay = 0) {
    // FIXME: calculate an actual time, not just using the delay, and make sure the indexes are 
    // in the right order
    return _simulation._baby_cl[sp][delay][_cell];
  }
};

#endif // CONTEXT_HPP
