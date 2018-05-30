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


// char fill, a lot like zfill but with any char
string cfill(string prFillMe, const char& pcfFillWith, const int& pcfFillLen);
// adds the "_####" right before file extension of a file name
string file_add_num(string prFileName, const string& pcfFillPrefix,
        const char& pcfFillWith, const int& pcfFillMe,
        const int& pcfFillLen, const string& pcfFillAt);

// converts comma seperated list of specie names to specie_vec
specie_vec str_to_species(std::string pcfSpecies);

#endif
