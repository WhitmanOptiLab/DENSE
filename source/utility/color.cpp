#include "style.hpp"


namespace style {

  namespace {
    Mode mode = Mode::force;
  }

  std::string apply (Color c) {
    return mode == Mode::force ? std::string("\x1b[3") + get_color_code(c) + 'm' : "";
  }

  std::string reset () {
    return mode == Mode::force ? "\x1b[0m" : "";
  }

  void configure(Mode mode) noexcept {
    style::mode = mode;
  }
}
