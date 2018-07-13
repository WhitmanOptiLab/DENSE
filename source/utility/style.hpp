// Struct for command line argument color

#ifndef UTIL_COLOR_HPP
#define UTIL_COLOR_HPP

#include <string>

namespace style {

  enum class Color : int_fast8_t {
    black, red, green, yellow, blue, magenta, cyan, white, initial
  };

  constexpr char get_color_code(Color color) noexcept {
    return "012345679"[static_cast<int_fast8_t>(color)];
  }

  std::string apply (Color c);

  std::string reset();

  enum class Mode : int_fast8_t { disable, detect, force };

  void configure(Mode) noexcept;

}

#endif // COLOR_HPP
