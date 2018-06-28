#include "utility/numerics.hpp"

#include <limits>
#include <stdexcept>

namespace dense {
namespace numeric_literals {

    constexpr dense::Real operator"" _R (long double value) noexcept {
      return static_cast<dense::Real>(value);
    }

    constexpr dense::Natural operator"" _N (long long unsigned value) noexcept {
      return value <= std::numeric_limits<dense::Natural>::max() ?
        static_cast<dense::Natural>(value) :
        throw std::logic_error("dense::Natural numeric literals must not overflow");
    }

    constexpr dense::Whole operator"" _Z (long long unsigned value) noexcept {
      return value <= std::numeric_limits<dense::Whole>::max() ?
        static_cast<dense::Whole>(value) :
        throw std::logic_error("dense::Whole numeric literals must not overflow");
    }

} }
