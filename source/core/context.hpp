// A context defines a locale in which reactions take place and species
//   reside
#ifndef CORE_CONTEXT_HPP
#define CORE_CONTEXT_HPP

#include "specie.hpp"
#include "util/common_utils.hpp"

class ContextBase {
  //FIXME - want to make this private at some point
 public:
  IF_CUDA(__host__ __device__)
  virtual Real getCon(specie_id sp) const = 0;
  IF_CUDA(__host__ __device__)
  virtual void advance() = 0;
  IF_CUDA(__host__ __device__)
  virtual bool isValid() const = 0;
  IF_CUDA(__host__ __device__)
  virtual void set(int c) = 0;
};

#endif // CONTEXT_HPP
