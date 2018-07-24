#ifndef CORE_SPECIE_HPP
#define CORE_SPECIE_HPP

#include <string>

// Try to ensure that enum values are assigned in the range [0, size)
// and not user-determined as in SPECIE(example = 10)


enum Species {
  #define SPECIE(name) name,
  #define CRITICAL_SPECIE(name) name,
  #include "specie_list.hpp"
  #undef CRITICAL_SPECIE
  #undef SPECIE
  size
};

const std::string specie_str[Species::size] = {
  #define SPECIE(name) #name,
  #include "specie_list.hpp"
  #undef SPECIE
};

constexpr auto NUM_SPECIES = Species::size;
using specie_id = Species;

enum Critical_Species_ID {
#define SPECIE(name)
#define CRITICAL_SPECIE(name) rcrit_##name,
#include "specie_list.hpp"
#undef SPECIE
#undef CRITICAL_SPECIE
  NUM_CRITICAL_SPECIES
};

using critspecie_id = Critical_Species_ID;

#endif

