#ifndef SIM_SET_CUDA_HPP
#define SIM_SET_CUDA_HPP

#include "set.hpp"
#include "determ/determ_cuda.hpp"
#include <vector>
#include <array>
#include <iostream>


/* simulation contains simulation data, partially taken from input_params and partially derived from other information
	notes:
 There should be only one instance of simulation at any time.
	todo:
 */

class simulation_set_cuda {

 public:

    Real time_total;

    void param_set_init(Real* paramset){
        _ps->getArray()[rcrit_pd + NUM_REACTIONS + NUM_DELAY_REACTIONS] = paramset[44];
        _ps->getArray()[rcrit_ph11 + NUM_REACTIONS + NUM_DELAY_REACTIONS] = paramset[42];
        _ps->getArray()[rcrit_ph713 + NUM_REACTIONS + NUM_DELAY_REACTIONS] = paramset[43];

        _ps->getArray()[dreact_mh1_synthesis + NUM_DELAY_REACTIONS] = paramset[34];
        _ps->getArray()[dreact_mh7_synthesis + NUM_DELAY_REACTIONS] = paramset[35];
        _ps->getArray()[dreact_mh13_synthesis + NUM_DELAY_REACTIONS] = paramset[36];
        _ps->getArray()[dreact_md_synthesis + NUM_DELAY_REACTIONS] = paramset[37];
        _ps->getArray()[dreact_ph1_synthesis + NUM_DELAY_REACTIONS] = paramset[38];
        _ps->getArray()[dreact_ph7_synthesis + NUM_DELAY_REACTIONS] = paramset[39];
        _ps->getArray()[dreact_ph13_synthesis + NUM_DELAY_REACTIONS] = paramset[40];
        _ps->getArray()[dreact_pd_synthesis + NUM_DELAY_REACTIONS] = paramset[41];

        _ps->getArray()[mh1_synthesis] = paramset[0];
        _ps->getArray()[mh7_synthesis] = paramset[1];
        _ps->getArray()[mh13_synthesis] = paramset[2];
        _ps->getArray()[md_synthesis] = paramset[3];

        _ps->getArray()[mh1_degradation] = paramset[4];
        _ps->getArray()[mh7_degradation] = paramset[5];
        _ps->getArray()[mh13_degradation] = paramset[6];
        _ps->getArray()[md_degradation] = paramset[7];

        _ps->getArray()[ph1_synthesis] = paramset[8];
        _ps->getArray()[ph7_synthesis] = paramset[9];
        _ps->getArray()[ph13_synthesis] = paramset[10];
        _ps->getArray()[pd_synthesis] = paramset[11];

        _ps->getArray()[ph1_degradation] = paramset[12];
        _ps->getArray()[ph7_degradation] = paramset[13];
        _ps->getArray()[ph13_degradation] = paramset[14];
        _ps->getArray()[pd_degradation] = paramset[15];

        _ps->getArray()[ph11_association] = paramset[16];
        _ps->getArray()[ph17_association] = paramset[17];
        _ps->getArray()[ph113_association] = paramset[18];
        _ps->getArray()[ph77_association] = paramset[19];
        _ps->getArray()[ph713_association] = paramset[20];
        _ps->getArray()[ph1313_association] = paramset[21];

        _ps->getArray()[ph11_dissociation] = paramset[22];
        _ps->getArray()[ph17_dissociation] = paramset[23];
        _ps->getArray()[ph113_dissociation] = paramset[24];
        _ps->getArray()[ph77_dissociation] = paramset[25];
        _ps->getArray()[ph713_dissociation] = paramset[26];
        _ps->getArray()[ph1313_dissociation] = paramset[27];

        _ps->getArray()[ph11_degradation] = paramset[28];
        _ps->getArray()[ph17_degradation] = paramset[29];
        _ps->getArray()[ph113_degradation] = paramset[30];
        _ps->getArray()[ph77_degradation] = paramset[31];
        _ps->getArray()[ph713_degradation] = paramset[32];
        _ps->getArray()[ph1313_degradation] = paramset[33];
    }



    simulation_set_cuda(int num_param, bool using_gradients, bool using_perturb, Real* paramset, int cell_total, int total_width, Real step_size, Real analysis_interval, Real sim_time) : _num_sets(num_param) {
        cudaMallocManaged(&_ps, sizeof(Parameter_Set));
        cudaMallocManaged(&_m, sizeof(dense::model));
        new(_m) dense::model();
	time_total = sim_time;
        cudaMallocManaged(&_sim_set, sizeof(simulation_cuda)*_num_sets);
        param_set_init(paramset);
        for (int i = 0; i < _num_sets; i++){
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
    dense::model* _m;
    Parameter_Set* _ps;
    simulation_cuda* _sim_set;
    int _num_sets;
};
#endif
