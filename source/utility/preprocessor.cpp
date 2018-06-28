#include "utility/preprocessor.hpp"

#include <iostream>

namespace preprocessor {

  /// Insert the characters of a text view into a `std::ostream`.
  inline std::ostream & operator<< (std::ostream & out, Text const& text) {
    return out.write(text.data(), text.size());
  }

}
