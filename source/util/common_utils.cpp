#include "common_utils.hpp"

string file_add_num(string prFileName, const string& pcfFillPrefix,
        const char& pcfFillWith, const int& pcfFillMe,
        const int& pcfFillLen, const string& pcfFillAt)
{
    prFileName = prFileName.substr(0,
        prFileName.find_last_of(pcfFillAt)) + pcfFillPrefix +
        cfill(to_string(pcfFillMe), pcfFillWith, pcfFillLen) +
        prFileName.substr(prFileName.find_last_of(pcfFillAt));
    return prFileName;
}

string cfill(string prFileName, const char& pcfFillWith, const int& pcfFillLen)
{
    while (prFileName.length() < pcfFillLen)
    {
        prFileName = pcfFillWith + prFileName;
    }

    return prFileName;
}

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
