#ifndef SIMULATION_SET_HPP
#define SIMULATION_SET_HPP

#include "param_set.hpp"
#include "model.hpp"
#include "cell_param.hpp"
#include "reaction.hpp"
#include "concentration_level.hpp"
#include "baby_cl.hpp"
#include "simulation_base.hpp"
#include "simulation_stoch.hpp"
#include "simulation_determ.hpp"
#include "csvr_param.hpp"
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
    vector<param_set> _ps;
    vector<simulation_base*> _sim_set;
    
    
    simulation_set(bool using_gradients, bool using_perturb, const string &param_file, int cell_total, int total_width, RATETYPE step_size, RATETYPE analysis_interval, RATETYPE sim_time, int seed) :
        _m(using_gradients, using_perturb)
    {
        csvr_param csvrp(param_file);
        // Setup only if param_file actually exists
        
        if (csvrp.is_open())
        {
            // Allocate space based on amount of sets in file
            iSetCount = csvrp.get_total();
            _ps.reserve(iSetCount);
            _sim_set.reserve(iSetCount);
            
            // For each set, load data to _ps and _sim_set
            for (unsigned int i=0; i<iSetCount; i++)
            {
                cout << "Initializing Set " << i << endl;
                _ps.push_back(csvrp.get_next());
                
                // When init'ing a sim_set<sim_base>, have step_size be = to 0.0 so that sim_set can emplace_back correctly
                if (step_size == 0.0)
                {
                    _sim_set.push_back(
                            new simulation_stoch(_m, _ps[i], cell_total, total_width,
                                analysis_interval, sim_time, seed));
                }
                else
                {
                    _sim_set.push_back(
                            new simulation_determ(_m, _ps[i], cell_total, total_width,
                                step_size, analysis_interval, sim_time));
                }

                _sim_set[i]->initialize();
            }
        }
    }
    
    void simulate_sets(){
        for (int i=0; i<_sim_set.size(); i++){
            cout << "Simulating Set " << i << endl;
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

