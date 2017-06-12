// A context defines a locale in which reactions take place and species 
//   reside
#ifndef CONTEXT_HPP
#define CONTEXT_HPP

#include "specie.hpp"
#include "concentration_level.hpp"


class ContextBase {
  //FIXME - want to make this private at some point
 public:
  CPUGPU_FUNC
  virtual RATETYPE getCon(specie_id sp) const;
  CPUGPU_FUNC
  virtual void advance();
  CPUGPU_FUNC
  virtual bool isValid() const;
};

#endif // CONTEXT_HPP
