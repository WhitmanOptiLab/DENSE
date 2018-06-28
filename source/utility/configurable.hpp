#ifndef DENSE_CONFIGURABLE_HPP
#define DENSE_CONFIGURABLE_HPP

#include "utility/preprocessor.hpp"

#include <type_traits>

namespace {
  constexpr bool is_benign_typename(preprocessor::Text const& text) {
    return text.is_subset_of(
      "ABCDEFGHIJKLMNOPQRSTUVWXYZ" "0123456789"
      "abcdefghijklmnopqrstuvwxyz" "_: *<>[]");
  }
}

#define DENSE_CONFIGURABLE_EXPAND(NAME)\
  PREPROCESSOR_GET_EXPANDED_TEXT(DENSE_CONFIGURABLE_##NAME)

#define DENSE_CONFIGURABLE_CITE(NAME)\
  #NAME " (aka \"" DENSE_CONFIGURABLE_EXPAND(NAME) "\")"

#define DENSE_CONFIGURABLE_TYPE_ALIAS(NAME)\
  static_assert(is_benign_typename(DENSE_CONFIGURABLE_EXPAND(NAME)),\
    "dense::" DENSE_CONFIGURABLE_CITE(NAME) " must be a typename containing"\
    " only benign characters: alphanumerics, underscores, colons, spaces,"\
    " asterisks, and square or angle brackets");\
  using NAME = DENSE_CONFIGURABLE_##NAME;\
  static_assert(true, "") // Require ';' and ensure TYPE does not leak

namespace dense {
namespace configurable {

  #ifndef DENSE_CONFIGURABLE_Real
  #define DENSE_CONFIGURABLE_Real double
  #endif

  DENSE_CONFIGURABLE_TYPE_ALIAS(Real);
  static_assert(std::is_floating_point<Real>() &&
    std::is_same<Real, std::remove_cv<Real>::type>(),
    "dense::" DENSE_CONFIGURABLE_CITE(Real) " must be a valid CV-unqualified"
    "floating-point type");

  #undef DENSE_CONFIGURABLE_Real

} }

#undef DENSE_CONFIGURABLE_TYPE_ALIAS
#undef DENSE_CONFIGURABLE_CITE
#undef DENSE_CONFIGURABLE_EXPAND

#endif
