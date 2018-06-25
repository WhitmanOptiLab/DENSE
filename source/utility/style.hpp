// Struct for command line argument color

#ifndef UTIL_COLOR_HPP
#define UTIL_COLOR_HPP

#include <string>

namespace style {

  enum class Color : char {
    black = '0', red, green, yellow, blue, magenta, cyan, white, initial = '9'
  };

  void enable (bool value = true) noexcept;

  void disable() noexcept;

  std::string apply (Color c);

  std::string reset();

}

using Color = style::Color;

#endif // COLOR_HPP
