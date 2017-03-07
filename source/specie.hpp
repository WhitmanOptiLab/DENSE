#ifndef SPECIE_HPP
#define SPECIE_HPP

enum specie_id {
#define SPECIE(name) name, 
#include "specie_list.hpp"
#undef SPECIE
  NUM_SPECIES  //And a terminal marker so that we know how many there are
};

enum critspecie_id {
#define SPECIE(name) 
#define CRITICAL_SPECIE(name) rcrit_##name,
#include "specie_list.hpp"
#undef SPECIE
#undef CRITICAL_SPECIE
  NUM_CRITICAL_SPECIES  //And a terminal marker so that we know how many there are
};
#endif

