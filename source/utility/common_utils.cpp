#include "common_utils.hpp"

#include "utility/style.hpp"
#include "io/csvr.hpp"
#include "core/reaction.hpp"

using style::Color;

#include <algorithm>
#include <iostream>
#include <cmath>

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


Real* parse_perturbations(std::string const& pcfPertFileName) {
      Real* factors_pert = nullptr;
      Real do_global_pert_val = strtold(pcfPertFileName.c_str(), nullptr);
      bool do_global_pert = (do_global_pert_val != 0.0);

      if (pcfPertFileName.size() > 0)
      {
          // Try loading file, suppress warning if string can be
          //   read as Real
          csvr perturbFile(pcfPertFileName, do_global_pert);
          if (perturbFile.is_open() || do_global_pert)
          {
              factors_pert = new Real[NUM_REACTIONS];

              // pert factor to be added to array
              Real tPert = 0.0;
              for (int i = 0; i < NUM_REACTIONS; i++)
              {
                  // Perturb default (0.0 if argument was not a Real)
                  // Prevents crashes in case perturbation parsing fails
                  factors_pert[i] = do_global_pert_val;

                  if (!do_global_pert)
                  {
                      if (perturbFile.get_next(&tPert))
                      {
                          factors_pert[i] = tPert;
                          tPert = 0.0;
                      }
                      else
                      {
                          // Error: Invalid number of filled cells
                          std::cout << style::apply(Color::red) <<
                              "CSV perturbations parsing failed. Ran out "
                              "of cells to read upon reaching reaction \""
                              << reaction_str[i] << "\"." <<
                              style::reset() << '\n';
                      }
                  }
              }
          }
      }
      return factors_pert;
    }

    Real** parse_gradients(std::string const& pcfGradFileName, int total_width) {
      Real** factors_grad = nullptr;
      if (pcfGradFileName.size() > 0) {
          csvr gradientFile(pcfGradFileName);
          if (gradientFile.is_open())
          {
              factors_grad = new Real*[NUM_REACTIONS];
              // gradient width index start, gradient width index end,
              //   gradient low bound, gradient high bound,
              //   gradient slope
              Real tGradX1 = 0.0, tGradX2 = 0.0,
                       tGradY1 = 0.0, tGradY2 = 0.0, tGradM = 0.0;
              for (std::size_t i = 0; i < NUM_REACTIONS; i++)
              {
                  // Gradient defaults
                  // Helps prevent crashes in case gradient parsing fails
                  factors_grad[i] = nullptr;

                  // Read all tGrad--s
                  if (gradientFile.get_next(&tGradX1) &&
                      gradientFile.get_next(&tGradY1) &&
                      gradientFile.get_next(&tGradX2) &&
                      gradientFile.get_next(&tGradY2) )
                  {
                      if (tGradX1 >= 0 && tGradX2 <= total_width)
                      {
                          // If equal, more than likely, user does not
                          //   want to enable gradients for this specie
                          if (tGradX1!=tGradX2)
                          {
                              factors_grad[i] =
                                  new Real[total_width];
                              tGradM = (tGradY2 - tGradY1) /
                                  (tGradX2 - tGradX1);

                              for (int j = std::round(tGradX1);
                                      j <= std::round(tGradX2); j++)
                              {
                                  factors_grad[i][j] = tGradY1;
                                  tGradY1 += tGradM;
                              }
                          }
                      }
                      else
                      {
                          // Error: Invalid numbers in cells
                          std::cout << style::apply(Color::red) <<
                              "CSV gradients parsing failed. "
                              "Invalid grad_x1 and/or grad_x2 "
                              "setting(s) for reaction \"" <<
                              reaction_str[i] << "\"." <<
                              style::reset() << '\n';
                      }
                  }
                  else
                  {
                      // Error: Invalid number of filled cells
                      std::cout << style::apply(Color::red) <<
                          "CSV gradients parsing failed. "
                          "Ran out of cells to read upon "
                          "reaching reaction \"" << reaction_str[i] <<
                          "\"." << style::reset() << '\n';
                  }
              }
          }
      }
      return factors_grad;
    }
