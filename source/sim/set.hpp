#ifndef SIM_SET_HPP
#define SIM_SET_HPP

#include "utility/common_utils.hpp"
#include "core/parameter_set.hpp"
#include "core/model.hpp"
#include "cell_param.hpp"
#include "core/reaction.hpp"
#include "base.hpp"
#include "stoch/stoch.hpp"
#include "determ/determ.hpp"
#include "io/csvr.hpp"
#include "utility/style.hpp"

#include <vector>
#include <array>
#include <iostream>


/* simulation contains simulation data, partially taken from input_params and partially derived from other information
	notes:
 There should be only one instance of simulation at any time.
	todo:
 */


class Simulation_Set {

  public:
    std::vector<Parameter_Set> const& _ps;
    std::vector<Simulation*> _sim_set;
    //setting up model
    model _m;
    Real total_time;
    Real* factors_pert;
    Real** factors_grad;


    Simulation_Set(std::vector<Parameter_Set> const& params, std::string const& pcfGradFileName, std::string const& pcfPertFileName, int cell_total, int total_width, Real step_size, Real analysis_interval, Real sim_time, int seed) :
        _ps(params), factors_pert(nullptr), factors_grad(nullptr)
    {

            iSetCount = _ps.size();
            _sim_set.reserve(iSetCount);



            // PREPARE PERT AND GRAD FILES
            {
                Real do_global_pert_val = strtold(pcfPertFileName.c_str(), 0);
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

                if (pcfGradFileName.size() > 0)
                {
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
                            factors_grad[i] = 0;

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
            } // End prepare pert and grad files


            // For each set, load data to _ps and _sim_set
            for (std::size_t i = 0; i < _ps.size(); i++)
            {
                // When init'ing a sim_set<sim_base>, have step_size be = to 0.0 so that sim_set can emplace_back correctly
                if (step_size == 0.0)
                {
                    _sim_set.push_back(
                            new Stochastic_Simulation(_m, _ps[i], factors_pert,
                                factors_grad, cell_total, total_width,
                                analysis_interval, sim_time, seed));
                }
                else
                {
                    _sim_set.push_back(
                            new Deterministic_Simulation(_m, _ps[i], factors_pert,
                                factors_grad, cell_total, total_width,
                                step_size, analysis_interval, sim_time));
                }

                _sim_set[i]->initialize();
            }
    }

    void simulate_sets(){
        for (auto & set : _sim_set) {
            set->simulate();
        }
    }

    ~Simulation_Set()
    {

    }

    const unsigned int& getSetCount() const
    {
        return iSetCount;
    }

private:
    unsigned int iSetCount;
};
#endif
