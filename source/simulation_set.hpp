#ifndef SIMULATION_SET_HPP
#define SIMULATION_SET_HPP

#include "param_set.hpp"
#include "model.hpp"
#include "cell_param.hpp"
#include "reaction.hpp"
#include "concentration_level.hpp"
#include "baby_cl.hpp"
#include "simulation.hpp"
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
    param_set _ps;
    RATETYPE total_time;
    vector<simulation> _sim_set;

    void param_set_init(RATETYPE* paramset){
        _ps._critical_values[rcrit_pd] = paramset[44];
        _ps._critical_values[rcrit_ph11] = paramset[42];
        _ps._critical_values[rcrit_ph713] = paramset[43];
        
        _ps._delay_sets[dreact_mh1_synthesis] = paramset[34];
        _ps._delay_sets[dreact_mh7_synthesis] = paramset[35];
        _ps._delay_sets[dreact_mh13_synthesis] = paramset[36];
        _ps._delay_sets[dreact_md_synthesis] = paramset[37];
        _ps._delay_sets[dreact_ph1_synthesis] = paramset[38];
        _ps._delay_sets[dreact_ph7_synthesis] = paramset[39];
        _ps._delay_sets[dreact_ph13_synthesis] = paramset[40];
        _ps._delay_sets[dreact_pd_synthesis] = paramset[41];
        
        _ps._rates_base[mh1_synthesis] = paramset[0];
        _ps._rates_base[mh7_synthesis] = paramset[1];
        _ps._rates_base[mh13_synthesis] = paramset[2];
        _ps._rates_base[md_synthesis] = paramset[3];
        
        _ps._rates_base[mh1_degradation] = paramset[4];
        _ps._rates_base[mh7_degradation] = paramset[5];
        _ps._rates_base[mh13_degradation] = paramset[6];
        _ps._rates_base[md_degradation] = paramset[7];
        
        _ps._rates_base[ph1_synthesis] = paramset[8];
        _ps._rates_base[ph7_synthesis] = paramset[9];
        _ps._rates_base[ph13_synthesis] = paramset[10];
        _ps._rates_base[pd_synthesis] = paramset[11];
        
        _ps._rates_base[ph1_degradation] = paramset[12];
        _ps._rates_base[ph7_degradation] = paramset[13];
        _ps._rates_base[ph13_degradation] = paramset[14];
        _ps._rates_base[pd_degradation] = paramset[15];
        
        _ps._rates_base[ph11_association] = paramset[16];
        _ps._rates_base[ph17_association] = paramset[17];
        _ps._rates_base[ph113_association] = paramset[18];
        _ps._rates_base[ph77_association] = paramset[19];
        _ps._rates_base[ph713_association] = paramset[20];
        _ps._rates_base[ph1313_association] = paramset[21];
        
        _ps._rates_base[ph11_dissociation] = paramset[22];
        _ps._rates_base[ph17_dissociation] = paramset[23];
        _ps._rates_base[ph113_dissociation] = paramset[24];
        _ps._rates_base[ph77_dissociation] = paramset[25];
        _ps._rates_base[ph713_dissociation] = paramset[26];
        _ps._rates_base[ph1313_dissociation] = paramset[27];

        _ps._rates_base[ph11_degradation] = paramset[28];
        _ps._rates_base[ph17_degradation] = paramset[29];
        _ps._rates_base[ph113_degradation] = paramset[30];
        _ps._rates_base[ph77_degradation] = paramset[31];
        _ps._rates_base[ph713_degradation] = paramset[32];
        _ps._rates_base[ph1313_degradation] = paramset[33];
    }
    
    

    simulation_set(int num_param, bool using_gradients, bool using_perturb, RATETYPE* paramset, int cell_total, int total_width, RATETYPE step_size, RATETYPE analysis_interval, RATETYPE sim_time) :
      _m(using_gradients, using_perturb) {
        _sim_set.reserve(num_param);
        param_set_init(paramset);
	total_time = sim_time;
        for (int i=0; i< num_param; i++){
          _sim_set.emplace_back(_m, _ps, cell_total, total_width, step_size,analysis_interval,sim_time);
          _sim_set[i].initialize();
        }
    }
    
    void simulate_sets(){
        for (int i=0; i<_sim_set.size(); i++){
            _sim_set[i].simulate();
        }
    }
};
#endif

