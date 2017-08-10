#ifndef UTIL_COMMON_UTILS_HPP
#define UTIL_COMMON_UTILS_HPP

#ifdef __CUDACC__
#define CPUGPU_FUNC __host__ __device__
#else
#define CPUGPU_FUNC
#endif

#ifdef __CUDACC__
#define STATIC_VAR __managed__
#else
#define STATIC_VAR
#endif

#include "core/specie.hpp"
#include <string>
using namespace std;

typedef float RATETYPE;

template<class TYPE, int SIZE>
class CPUGPU_TempArray {
  TYPE array[SIZE];
 public:
  CPUGPU_FUNC
  TYPE& operator[](int i) { return array[i]; }
  CPUGPU_FUNC
  const TYPE& operator[](int i) const { return array[i]; }
};

#define WRAP(x, y) ((x) + (y)) % (y)

// char fill, a lot like zfill but with any char
string cfill(string prFillMe, const char& pcfFillWith, const int& pcfFillLen);
// adds the "_####" right before file extension of a file name
string file_add_num(string prFileName, const string& pcfFillPrefix,
        const char& pcfFillWith, const int& pcfFillMe,
        const int& pcfFillLen, const string& pcfFillAt);

// converts comma seperated list of specie names to specie_vec
specie_vec str_to_species(string pcfSpecies);


#endif
