// A context defines a locale in which reactions take place and species 
//   reside
#ifndef CONTEXT_HPP
#define CONTEXT_HPP

#include "specie.hpp"
#include "concentration_level.hpp"
class Context {
  //FIXME - want to make this private at some point
 public:
  const int _cell;
  const simulation& _simulation;
  double _avg;
  Context(const simulation& sim, int cell) : _simulation(sim),_cell(cell) { }
    void calculateNeighbourAvg(specie_id sp);
    void updateCon(Concentration_level& cl, double rates[]);
    double* calculateRates();
};

#endif // CONTEXT_HPP
