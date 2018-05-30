#include "common_utils.hpp"

std::string file_add_num (
  std::string file_name, std::string const& prefix,
  char padding, unsigned file_no,
  std::size_t padded_size, std::string const& extension_sep)
{
  auto padded_file_no = left_pad(std::to_string(file_no), padded_size, padding);
  auto before_extension_sep = std::min(file_name.find_last_of(extension_sep), file_name.size());
  file_name.insert(before_extension_sep, prefix + padded_file_no);
  return file_name;
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
