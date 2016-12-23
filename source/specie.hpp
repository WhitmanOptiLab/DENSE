#ifndef SPECIE_HPP
#define SPECIE_HPP

#include "specie_list.hpp"

enum specie_id {
#define SPECIE(name) name, 
LIST_OF_SPECIES
#undef SPECIE
  NUM_SPECIES  //And a terminal marker so that we know how many there are
};

#endif

