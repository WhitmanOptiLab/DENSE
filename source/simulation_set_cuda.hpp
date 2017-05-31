#ifndef SIMULATION_SET_CUDA_HPP
#define SIMULATION_SET_CUDA_HPP

#include "simulation_set.hpp"
#include "simulation_cuda.hpp"
#include <vector>
#include <array>
using namespace std;

/* simulation contains simulation data, partially taken from input_params and partially derived from other information
	notes:
 There should be only one instance of simulation at any time.
	todo:
 */

class simulation_set_cuda {
    
 public:
    
    RATETYPE time_total;    

    void param_set_init(RATETYPE* paramset){
        _ps->_critical_values[rcrit_pd] = paramset[44];
        _ps->_critical_values[rcrit_ph11] = paramset[42];
        _ps->_critical_values[rcrit_ph713] = paramset[43];
        
        _ps->_delay_sets[dreact_mh1_synthesis] = paramset[34];
        _ps->_delay_sets[dreact_mh7_synthesis] = paramset[35];
        _ps->_delay_sets[dreact_mh13_synthesis] = paramset[36];
        _ps->_delay_sets[dreact_md_synthesis] = paramset[37];
        _ps->_delay_sets[dreact_ph1_synthesis] = paramset[38];
        _ps->_delay_sets[dreact_ph7_synthesis] = paramset[39];
        _ps->_delay_sets[dreact_ph13_synthesis] = paramset[40];
        _ps->_delay_sets[dreact_pd_synthesis] = paramset[41];
        
        _ps->_rates_base[mh1_synthesis] = paramset[0];
        _ps->_rates_base[mh7_synthesis] = paramset[1];
        _ps->_rates_base[mh13_synthesis] = paramset[2];
        _ps->_rates_base[md_synthesis] = paramset[3];
        
        _ps->_rates_base[mh1_degradation] = paramset[4];
        _ps->_rates_base[mh7_degradation] = paramset[5];
        _ps->_rates_base[mh13_degradation] = paramset[6];
        _ps->_rates_base[md_degradation] = paramset[7];
        
        _ps->_rates_base[ph1_synthesis] = paramset[8];
        _ps->_rates_base[ph7_synthesis] = paramset[9];
        _ps->_rates_base[ph13_synthesis] = paramset[10];
        _ps->_rates_base[pd_synthesis] = paramset[11];
        
        _ps->_rates_base[ph1_degradation] = paramset[12];
        _ps->_rates_base[ph7_degradation] = paramset[13];
        _ps->_rates_base[ph13_degradation] = paramset[14];
        _ps->_rates_base[pd_degradation] = paramset[15];
        
        _ps->_rates_base[ph11_association] = paramset[16];
        _ps->_rates_base[ph17_association] = paramset[17];
        _ps->_rates_base[ph113_association] = paramset[18];
        _ps->_rates_base[ph77_association] = paramset[19];
        _ps->_rates_base[ph713_association] = paramset[20];
        _ps->_rates_base[ph1313_association] = paramset[21];
        
        _ps->_rates_base[ph11_dissociation] = paramset[22];
        _ps->_rates_base[ph17_dissociation] = paramset[23];
        _ps->_rates_base[ph113_dissociation] = paramset[24];
        _ps->_rates_base[ph77_dissociation] = paramset[25];
        _ps->_rates_base[ph713_dissociation] = paramset[26];
        _ps->_rates_base[ph1313_dissociation] = paramset[27];

        _ps->_rates_base[ph11_degradation] = paramset[28];
        _ps->_rates_base[ph17_degradation] = paramset[29];
        _ps->_rates_base[ph113_degradation] = paramset[30];
        _ps->_rates_base[ph77_degradation] = paramset[31];
        _ps->_rates_base[ph713_degradation] = paramset[32];
        _ps->_rates_base[ph1313_degradation] = paramset[33];
    }
    
    

    simulation_set_cuda(int num_param, bool using_gradients, bool using_perturb, RATETYPE* paramset, int cell_total, int total_width, RATETYPE step_size, RATETYPE analysis_interval, RATETYPE sim_time) : _num_sets(num_param) {
        cudaMallocManaged(&_ps, sizeof(param_set));
        cudaMallocManaged(&_m, sizeof(model));
        new(_m) model(false, false);
	time_total = sim_time;
        cudaMallocManaged(&_sim_set, sizeof(simulation_cuda)*_num_sets);
        param_set_init(paramset);
        for (int i=0; i< _num_sets; i++){
          new(&_sim_set[i]) simulation_cuda(*_m, *_ps, cell_total, total_width, step_size,analysis_interval,sim_time);
          _sim_set[i].initialize();
        }
    }

    void simulate_sets();

    ~simulation_set_cuda() {
      cudaFree(_ps);
      cudaFree(_m);
      for (int i = 0; i < _num_sets; i++) {
        _sim_set[i].~simulation_cuda();
      }
      cudaFree(_sim_set);
    }
 private:
    //setting up model
    model* _m;
    param_set* _ps;
    simulation_cuda* _sim_set;
    int _num_sets;
};
#endif
