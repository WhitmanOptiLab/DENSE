#ifndef UTIL_COMMON_UTILS_HPP
#define UTIL_COMMON_UTILS_HPP


#ifdef __CUDACC__
#define IF_CUDA(X) X
#else
#define IF_CUDA(X)
#endif

#define STATIC_VAR IF_CUDA(__managed__)

#include "core/specie.hpp"
// converts comma seperated list of specie names to specie_vec
specie_vec str_to_species(std::string pcfSpecies);

#include "Real.hpp"

template <typename ValueT, std::size_t Size>
class CUDA_Array {
 ValueT array[Size];
 public:
  IF_CUDA(__host__ __device__)
  ValueT & operator[] (std::size_t i) { return array[i]; };
  IF_CUDA(__host__ __device__)
  ValueT const& operator[] (std::size_t i) const { return array[i]; };
};

#endif
