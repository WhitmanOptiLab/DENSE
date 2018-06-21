#ifndef DENSE_UTILITY_CUDA_HPP
#define DENSE_UTILITY_CUDA_HPP

#include <system_error>
#include <type_traits>


#ifdef __CUDACC__
#define IF_CUDA(X) X
#define USING_CUDA
#else
#define IF_CUDA(X)
#endif

#define STATIC_VAR IF_CUDA(__managed__)

template <typename T, std::size_t size>
class CUDA_Array {

 public:

  IF_CUDA(__host__ __device__)
  T* begin() { return data_; }

  IF_CUDA(__host__ __device__)
  T const* begin() const { return data_; }

  IF_CUDA(__host__ __device__)
  T* end() { return data_ + size; }

  IF_CUDA(__host__ __device__)
  T const* end() const { return data_ + size; }

  IF_CUDA(__host__ __device__)
  T & operator[] (std::size_t i) {
    return data_[i];
  };

  IF_CUDA(__host__ __device__)
  T const& operator[] (std::size_t i) const {
    return data_[i];
  };

  private:

    T data_[size];

};

#ifdef USING_CUDA_NOPE


namespace cuda {

  /// Type alias for the CUDA error enumeration.
  /// \see [\c cudaError]
  ///   (https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__<!--
  ///   -->TYPES.html#group__CUDART__TYPES_1g3f51e3575c2178246db0a94a430e0038)
  using Error_Code = cudaError;

  namespace {

    class Error_Category : public std::error_category {

      public:

        char const* name() const noexcept override { return "cuda"; }

        std::string message(int value) const override {
          return cudaGetErrorString(static_cast<cuda::Error_Code>(value));
        }

    };

  }

  /// Get a reference to the CUDA error category singleton.
  ///  `std::error_category::name()` is overridden to return `"cuda"`.
  ///  `std::error_category::message(int)` is overridden to return a CUDA-specific
  ///   error string for the given `int` interpreted as a cuda::Error_Code.
  /// \see [\c cudaGetErrorString(cudaError)]
  ///   (https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__<!--
  ///   -->ERROR.html#group__CUDART__ERROR_1g4bc9e35a618dfd0877c29c8ee45148f1)
  std::error_category const& error_category() {
    static cuda::Error_Category error_category;
    return error_category;
  }

}

namespace std {

  /// Template specialization marking cuda::Error_Code eligible for conversion
  /// to `std::error_code`.
  template <>
  struct is_error_code_enum<cuda::Error_Code> : std::true_type {};

}

/// Convert cuda::Error_Code to `std::error_code`.
///   Not meant to be called directly; called via argument-dependent lookup
///   by the `std::error_code` constructor and assignment operator templates.
std::error_code make_error_code(cuda::Error_Code code) noexcept {
  return { static_cast<int>(code), cuda::error_category() };
}

#endif

#endif
