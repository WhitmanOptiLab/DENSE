#ifndef DENSE_UTILITY_CUDA_HPP
#define DENSE_UTILITY_CUDA_HPP

#include <system_error>
#include <type_traits>


#ifdef __CUDACC__
#define CUDA_QUALIFY(Q) Q
#define USING_CUDA
#else
#define CUDA_QUALIFY(Q)
#endif

#define IF_CUDA(X) CUDA_QUALIFY(X)
#define CUDA_HOST CUDA_QUALIFY(__host__)
#define CUDA_DEVICE CUDA_QUALIFY(__device__)
#define CUDA_KERNEL CUDA_QUALIFY(__global__)
#define CUDA_MANAGED CUDA_QUALIFY(__managed__)
#define STATIC_VAR CUDA_MANAGED

template <typename T, std::ptrdiff_t size>
class CUDA_Array {

 public:

  CUDA_HOST CUDA_DEVICE
  T* begin() { return data_; }

  CUDA_HOST CUDA_DEVICE
  T const* begin() const { return data_; }

  CUDA_HOST CUDA_DEVICE
  T* end() { return data_ + size; }

  CUDA_HOST CUDA_DEVICE
  T const* end() const { return data_ + size; }

  CUDA_HOST CUDA_DEVICE
  T & operator[] (std::ptrdiff_t i) {
    return data_[i];
  };

  CUDA_HOST CUDA_DEVICE
  T const& operator[] (std::ptrdiff_t i) const {
    return data_[i];
  };

  public:

    T data_[size];

};

namespace cuda {

  template <typename T>
  CUDA_HOST CUDA_DEVICE
  T min(T first, T second) {
    return first < second ? first : second;
  }

  template <typename T>
  CUDA_HOST CUDA_DEVICE
  T max(T first, T second) {
    return first > second ? first : second;
  }

}

#ifdef USING_CUDA


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
