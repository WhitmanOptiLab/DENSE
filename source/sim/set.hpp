#ifndef SIM_SET_HPP
#define SIM_SET_HPP

#include "core/param_set.hpp"
#include "core/model.hpp"
#include "cell_param.hpp"
#include "core/reaction.hpp"
#include "base.hpp"
#include "stoch/stoch.hpp"
#include "determ/determ.hpp"
#include "io/csvr_param.hpp"
#include "util/color.hpp"
#include <vector>
#include <array>
using namespace std;

/* simulation contains simulation data, partially taken from input_params and partially derived from other information
	notes:
 There should be only one instance of simulation at any time.
	todo:
 */


class simulation_set{
    
 public:
    //setting up model
    model _m;
   // param_set _ps;
    RATETYPE total_time;
    RATETYPE* factors_pert;
    RATETYPE** factors_grad;
    vector<param_set> _ps;
    vector<simulation_base*> _sim_set;
    
    
    simulation_set(const string &param_file, const string& pcfGradFileName, const string& pcfPertFileName, int cell_total, int total_width, RATETYPE step_size, RATETYPE analysis_interval, RATETYPE sim_time, int seed) :
        factors_pert(0), factors_grad(0)
    {
        csvr_param csvrp(param_file);
        // Setup only if param_file actually exists
        
        if (csvrp.is_open())
        {
            // Allocate space based on amount of sets in file
            iSetCount = csvrp.get_total();
            _ps.reserve(iSetCount);
            _sim_set.reserve(iSetCount);
            
            
            
            // PREPARE PERT AND GRAD FILES
            {
                RATETYPE do_global_pert_val = strtold(pcfPertFileName.c_str(), 0);
                bool do_global_pert = (do_global_pert_val != 0.0);
                
                if (pcfPertFileName.size() > 0)
                {
                    // Try loading file, suppress warning if string can be
                    //   read as RATETYPE
                    csvr perturbFile(pcfPertFileName, do_global_pert);
                    if (perturbFile.is_open() || do_global_pert)
                    {
                        factors_pert = new RATETYPE[NUM_REACTIONS];
                        
                        // pert factor to be added to array
                        RATETYPE tPert = 0.0;
                        for (int i=0; i<NUM_REACTIONS; i++)
                        {
                            // Perturb default (0.0 if argument was not a RATETYPE)
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
                                    cout << color::set(color::RED) <<
                                        "CSV perturbations parsing failed. Ran out "
                                        "of cells to read upon reaching reaction \""
                                        << reaction_str[i] << "\"." <<
                                        color::clear() << endl;
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
                        factors_grad = new RATETYPE*[NUM_REACTIONS];
                        // gradient width index start, gradient width index end,
                        //   gradient low bound, gradient high bound,
                        //   gradient slope
                        RATETYPE tGradX1 = 0.0, tGradX2 = 0.0,
                                 tGradY1 = 0.0, tGradY2 = 0.0, tGradM = 0.0;
                        for (int i = 0; i < NUM_REACTIONS; i++)
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
                                if (tGradX1>=0 && tGradX2<=total_width)
                                {
                                    // If equal, more than likely, user does not
                                    //   want to enable gradients for this specie
                                    if (tGradX1!=tGradX2)
                                    {
                                        factors_grad[i] =
                                            new RATETYPE[total_width];
                                        tGradM = (tGradY2 - tGradY1) /
                                            (tGradX2 - tGradX1);

                                        for (int j=round(tGradX1);
                                                j<=round(tGradX2); j++)
                                        {
                                            factors_grad[i][j] = tGradY1;
                                            tGradY1 += tGradM;
                                        }
                                    }
                                }
                                else
                                {
                                    // Error: Invalid numbers in cells
                                    cout << color::set(color::RED) <<
                                        "CSV gradients parsing failed. "
                                        "Invalid grad_x1 and/or grad_x2 "
                                        "setting(s) for reaction \"" <<
                                        reaction_str[i] << "\"." <<
                                        color::clear() << endl;
                                }
                            }
                            else
                            {
                                // Error: Invalid number of filled cells
                                cout << color::set(color::RED) <<
                                    "CSV gradients parsing failed. "
                                    "Ran out of cells to read upon "
                                    "reaching reaction \"" << reaction_str[i] <<
                                    "\"." << color::clear() << endl;
                            }
                        }
                    }
                }
            } // End prepare pert and grad files
            
            
            // For each set, load data to _ps and _sim_set
            for (unsigned int i=0; i<iSetCount; i++)
            {
                _ps.push_back(csvrp.get_next());
                
                // When init'ing a sim_set<sim_base>, have step_size be = to 0.0 so that sim_set can emplace_back correctly
                if (step_size == 0.0)
                {
                    _sim_set.push_back(
                            new simulation_stoch(_m, _ps[i], factors_pert,
                                factors_grad, cell_total, total_width,
                                analysis_interval, sim_time, seed));
                }
                else
                {
                    _sim_set.push_back(
                            new simulation_determ(_m, _ps[i], factors_pert,
                                factors_grad, cell_total, total_width,
                                step_size, analysis_interval, sim_time));
                }

                _sim_set[i]->initialize();
            }
        }
    }
    
    void simulate_sets(){
        for (int i=0; i<_sim_set.size(); i++){
            _sim_set[i]->simulate();
        }
    }
    
    ~simulation_set()
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

