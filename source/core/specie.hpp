#ifndef CORE_SPECIE_HPP
#define CORE_SPECIE_HPP

#include <string>
#include <vector>

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


const std::string specie_str[NUM_SPECIES] = {
    #define SPECIE(name) #name,
    #include "specie_list.hpp"
    #undef SPECIE
};

typedef std::vector<specie_id> specie_vec;

#endif

