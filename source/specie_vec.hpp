#ifndef SPECIE_VEC_HPP
#define SPECIE_VEC_HPP
#include "specie.hpp"

#include <string>
#include <vector>


const std::string specie_str[NUM_SPECIES] = {
    #define SPECIE(name) #name,
    #include "specie_list.hpp"
    #undef SPECIE
};

typedef std::vector<specie_id> specie_vec;

#endif
