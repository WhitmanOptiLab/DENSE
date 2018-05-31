#include "common_utils.hpp"

#include <algorithm>

specie_vec str_to_species(std::string pSpecies)
{
    pSpecies = "," + pSpecies + ",";
    pSpecies.erase(
            std::remove(pSpecies.begin(), pSpecies.end(), ' '),
            pSpecies.end() );

    specie_vec rVec;
    rVec.reserve(NUM_SPECIES);
    for (unsigned int i = 0; i < NUM_SPECIES; i++)
    {
        if (pSpecies.find(","+specie_str[i]+",") != std::string::npos ||
                pSpecies == ",," || pSpecies == ",all,")
        {
            rVec.push_back((specie_id) i);
        }
    }

    return rVec;
}
