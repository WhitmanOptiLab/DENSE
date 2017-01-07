// A context defines a locale in which reactions take place and species 
//   reside
#ifndef CONTEXT_HPP
#define CONTEXT_HPP

#include "specie.hpp"

template <class E>
class Context {
  //FIXME - want to make this private at some point
 public:
  const simulation& _simulation;
  Context(const const simulation& sim) : _simulation(sim) { }
};

#endif // CONTEXT_HPP
