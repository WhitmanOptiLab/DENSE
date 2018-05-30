#include "common_utils.hpp"

std::string file_add_num (
  std::string prFileName, std::string const& pcfFillPrefix,
  char pcfFillWith, unsigned pcfFillMe,
  std::size_t pcfFillLen, std::string const& pcfFillAt)
{
    prFileName = prFileName.substr(0,
        prFileName.find_last_of(pcfFillAt)) + pcfFillPrefix +
        left_pad(std::to_string(pcfFillMe), pcfFillWith, pcfFillLen) +
        prFileName.substr(prFileName.find_last_of(pcfFillAt));
    return prFileName;
};

std::string left_pad (std::string string, std::size_t min_size, char padding) {
  string.insert(string.begin(), min_size - std::min(min_size, string.size()), padding);
  return string;
};

#include <algorithm>

specie_vec str_to_species(std::string pSpecies)
{
    pSpecies = "," + pSpecies + ",";
    pSpecies.erase(
            std::remove(pSpecies.begin(), pSpecies.end(), ' '),
            pSpecies.end() );

    specie_vec rVec;
    rVec.reserve(NUM_SPECIES);
    for (unsigned int i=0; i<NUM_SPECIES; i++)
    {
        if (pSpecies.find(","+specie_str[i]+",") != std::string::npos ||
                pSpecies == ",," || pSpecies == ",all,")
        {
            rVec.push_back((specie_id) i);
        }
    }

    return rVec;
}
