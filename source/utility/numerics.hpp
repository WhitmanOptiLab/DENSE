#ifndef DENSE_NUMERICS_HPP
#define DENSE_NUMERICS_HPP

#include "utility/configurable.hpp"

#include <cstddef>

namespace dense {

  /// A signed integral type used to represent object and array indices.
  ///   Guaranteed to be large enough to store the size of an array of any type.
  using Index = std::ptrdiff_t;

  /// A *signed* integral type used to represent natural (non-negative) numbers.
  ///   Unsignedness is by convention; true unsigned types have the undesirable
  ///   requirement of wrapping on overflow (modular arithmetic).
  using Natural = int;

  /// A signed integral type used to represent whole numbers.
  using Whole = int;

  /// A configurable floating-point type used to represent real numbers.
  using dense::configurable::Real;

  namespace numeric_literals {

    constexpr dense::Real operator"" _R (long double value) noexcept;

    constexpr dense::Natural operator"" _N (long long unsigned value) noexcept;

    constexpr dense::Whole operator"" _Z (long long unsigned value) noexcept;

  }

}

using dense::Real;
using dense::Natural;
using RATETYPE = ::Real;

#endif
