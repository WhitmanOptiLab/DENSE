// A context defines a locale in which reactions take place and species 
//   reside
#ifndef CONTEXT_HPP
#define CONTEXT_HPP

#include "specie.hpp"
#include "common_utils.hpp"

class ContextBase {
  //FIXME - want to make this private at some point
 public:
  CPUGPU_FUNC
  virtual RATETYPE getCon(specie_id sp) const = 0;
  CPUGPU_FUNC
  virtual void advance() = 0;
  CPUGPU_FUNC
  virtual bool isValid() const = 0;
  CPUGPU_FUNC
  virtual void reset() = 0;
};

#endif // CONTEXT_HPP
