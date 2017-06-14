#ifndef SIMULATION_SET_HPP
#define SIMULATION_SET_HPP

#include "param_set.hpp"
#include "model.hpp"
#include "cell_param.hpp"
#include "reaction.hpp"
#include "concentration_level.hpp"
#include "baby_cl.hpp"
#include "simulation.hpp"
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
    vector<simulation> _sim_set;
    
    
    simulation_set(bool using_gradients, bool using_perturb, const string &param_file, int cell_total, int total_width, RATETYPE step_size, RATETYPE analysis_interval, RATETYPE sim_time) :
        _m(using_gradients, using_perturb)
    {
        csvr_param csvrp(param_file);
        // Setup only if param_file actually exists
    
        // Allocate space based on amount of sets in file
        unsigned int set_count = csvrp.get_total();
        _ps.reserve(set_count);
        _sim_set.reserve(set_count);
        
        // For each set, load data to _ps and _sim_set
        for (unsigned int i=0; i<set_count; i++)
        {
            _ps.push_back(csvrp.get_next());
            _sim_set.emplace_back(_m, _ps[i], cell_total, total_width, step_size, analysis_interval, sim_time);
            _sim_set[i].initialize();
        }
    }
    
    void simulate_sets(){
        for (int i=0; i<_sim_set.size(); i++){
            _sim_set[i].simulate();
        }
    }
    
    ~simulation_set()
    {
        
    }
};
#endif

