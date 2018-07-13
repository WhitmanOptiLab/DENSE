#ifndef UTIL_COMMON_UTILS_HPP
#define UTIL_COMMON_UTILS_HPP

#include "core/specie.hpp"
#include "utility/numerics.hpp"
// converts comma-separated list of specie names to specie_vec
specie_vec str_to_species(std::string pcfSpecies);
Real* parse_perturbations(std::string const& file_name);
Real** parse_gradients(std::string const& file_name, int width);

#endif
