#ifndef UTIL_COMMON_UTILS_HPP
#define UTIL_COMMON_UTILS_HPP

#include "core/specie.hpp"
#include "utility/numerics.hpp"
#include <memory>
#include <vector>

Species get_species_by_name (std::string const& name);

// converts comma-separated list of specie names to std::vector<Species>
std::vector<Species> str_to_species(std::string pcfSpecies);
Real* parse_perturbations(std::string const& file_name);
Real** parse_gradients(std::string const& file_name, int width);

namespace std14 {

  template <typename T, typename... Args>
  std::unique_ptr<T> make_unique(Args&&... args) {
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
  }

}

#endif
