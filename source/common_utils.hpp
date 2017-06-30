#ifndef COMMON_UTILS_HPP
#define COMMON_UTILS_HPP

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

typedef double RATETYPE;

template<class TYPE, int SIZE>
class CPUGPU_TempArray {
  TYPE array[SIZE];
 public:
  CPUGPU_FUNC
  TYPE& operator[](int i) { return array[i]; }
  CPUGPU_FUNC
  const TYPE& operator[](int i) const { return array[i]; }
};


#endif
