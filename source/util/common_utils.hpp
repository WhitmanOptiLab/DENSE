#ifndef UTIL_COMMON_UTILS_HPP
#define UTIL_COMMON_UTILS_HPP


#ifdef __CUDACC__
#define IF_CUDA(X) X
#else
#define IF_CUDA(X)
#endif

#define CPUGPU_FUNC IF_CUDA(__host__ __device__)

#define STATIC_VAR IF_CUDA(__managed__)

#include "core/specie.hpp"
#include <string>
using namespace std;

using Real = float;
using RATETYPE = Real;

template <typename ValueT, std::size_t Size>
class CPUGPU_TempArray {
 ValueT array[Size];
 public:
  CPUGPU_FUNC
  ValueT & operator[] (std::size_t i) { return array[i]; };
  CPUGPU_FUNC
  ValueT const& operator[] (std::size_t i) const { return array[i]; };
};


std::string left_pad (std::string string, std::size_t min_size, char padding = ' ');

// adds the "_####" right before file extension of a file name
std::string file_add_num (
  std::string file_name,
  std::string const& prefix,
  char padding, unsigned file_no,
  std::size_t padded_size,
  std::string const& extension_sep
);

// converts comma seperated list of specie names to specie_vec
specie_vec str_to_species(std::string pcfSpecies);

#endif
