#include "style.hpp"


namespace style {

  namespace {
    bool is_enabled = true;
  }

  void enable (bool value) noexcept {
    is_enabled = value;
  }

  void disable () noexcept {
    enable(false);
  }

  std::string apply (Color c) {
    return is_enabled ? std::string("\x1b[3") + static_cast<char>(c) + 'm' : "";
  }

  std::string reset () {
    return is_enabled ? "\x1b[0m" : "";
  }
}
