// A context defines a locale in which reactions take place and species 
//   reside
#ifndef CONTEXT_HPP
#define CONTEXT_HPP

template <class E>
class Context {
  //FIXME - want to make this private at some point
 public:
  std::vector<E> concentrations;
  Context() : concentrations(NUM_SPECIES) { }
};

#endif // CONTEXT_HPP
