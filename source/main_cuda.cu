#include "sim/set_cuda.hpp"
#include <iostream>

int main() {
    
    RATETYPE param_set[NUM_REACTIONS+NUM_DELAY_REACTIONS+NUM_CRITICAL_SPECIES] ={
        55.282,55.8188,44.856,35.3622,
        0.215387,0.37152,0.10447,0.453508,
        25.8443,40.6742,35.8334,54.2915,
        0.210157,0.269844,0.335617,0.282566,
        0.020873,0.002822,0.027891,0.027807,0.017196,0.020549,
        0.280481,0.256498,0.030745,0.296646,0.067092,0.195206,
        0.296753,0.324761,0.144681,0.253744,0.240119,0.205776,
        10.6427,9.15516,0,11.2572,1.72161,1.93494,0.805212,11.3695,
        713.625,740.349,201.173};
    
    RATETYPE analysis_interval = 100;
    RATETYPE sim_time = 60;
    simulation_set_cuda sim_set(400, false, false, param_set, 200, 50, 0.01,analysis_interval,sim_time);
    sim_set.simulate_sets();
}
