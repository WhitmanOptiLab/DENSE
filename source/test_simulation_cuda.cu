#include "simulation_cuda.hpp"
#include <iostream>

#define CPUGPU_ALLOC(type, var, ...) \
  type* var##_ptr; \
  cudaMallocManaged(&var##_ptr, sizeof(type)); \
  type& var = *(new(var##_ptr) type(__VA_ARGS__))

#define CPUGPU_DELETE(type, var) \
  var.~type(); \
  cudaFree(&var);
  


int main() {
    cudaSetDevice(0);
    //setting up model
    CPUGPU_ALLOC(model, m, false, false);
    
    //setting up param_set
    CPUGPU_ALLOC(param_set, ps);
    /*
     critical_values[NUM_CRITICAL_SPECIES] = {713.625,740.349,201.173};
     delay_sets[NUM_DELAY_REACTIONS] = {10.6427,9.15516,0,11.2572,1.72161,1.93494,0.805212,11.3695};
     rates_base[NUM_REACTIONS] = {
        55.282,55.8188,44.856,35.3622,
        0.215387,0.37152,0.10447,0.453508,
        25.8443,40.6742,35.8334,54.2915,
        0.210157,0.269844,0.335617,0.282566,
        0.020873,0.002822,0.027891,0.027807,0.017196,0.020549,
        0.280481,0.256498,0.030745,0.296646,0.067092,0.195206,
        0.296753,0.324761,0.144681,0.253744,0.240119,0.205776};
    */
     ps._critical_values[rcrit_pd] = 201.173;
     ps._critical_values[rcrit_ph11] = 713.625;
     ps._critical_values[rcrit_ph713] = 740.349;
    
     ps._delay_sets[dreact_mh1_synthesis] = 10.6427;
     ps._delay_sets[dreact_mh7_synthesis] = 9.15516;
     ps._delay_sets[dreact_mh13_synthesis] = 0;
     ps._delay_sets[dreact_md_synthesis] = 11.2572;
     ps._delay_sets[dreact_ph1_synthesis] = 1.72161;
     ps._delay_sets[dreact_ph7_synthesis] = 1.93494;
     ps._delay_sets[dreact_ph13_synthesis] = 0.805212;
     ps._delay_sets[dreact_pd_synthesis] = 11.3695;
    
     ps._rates_base[mh1_synthesis] = 55.282;
     ps._rates_base[mh7_synthesis] = 55.8188;
     ps._rates_base[mh13_synthesis] = 44.856;
     ps._rates_base[md_synthesis] = 35.3622;
    
     ps._rates_base[mh1_degradation] = 0.215387;
     ps._rates_base[mh7_degradation] = 0.37152;
     ps._rates_base[mh13_degradation] = 0.10447;
     ps._rates_base[md_degradation] = 0.453508;
    
     ps._rates_base[ph1_synthesis] = 25.8443;
     ps._rates_base[ph7_synthesis] = 40.6742;
     ps._rates_base[ph13_synthesis] = 35.8334;
     ps._rates_base[pd_synthesis] = 54.2915;
    
     ps._rates_base[ph1_degradation] = 0.210157;
     ps._rates_base[ph7_degradation] = 0.269844;
     ps._rates_base[ph13_degradation] = 0.335617;
     ps._rates_base[pd_degradation] = 0.282566;
    
     ps._rates_base[ph11_association] = 0.020873;
     ps._rates_base[ph17_association] = 0.002822;
     ps._rates_base[ph113_association] = 0.027891;
     ps._rates_base[ph77_association] = 0.027807;
     ps._rates_base[ph713_association] = 0.017196;
     ps._rates_base[ph1313_association] = 0.020549;
    
     ps._rates_base[ph11_dissociation] = 0.280481;
     ps._rates_base[ph17_dissociation] = 0.256498;
     ps._rates_base[ph113_dissociation] = 0.030745;
     ps._rates_base[ph77_dissociation] = 0.296646;
     ps._rates_base[ph713_dissociation] = 0.067092;
     ps._rates_base[ph1313_dissociation] = 0.195206;
    
     ps._rates_base[ph11_degradation] = 0.296753;
     ps._rates_base[ph17_degradation] = 0.324761;
     ps._rates_base[ph113_degradation] = 0.144681;
     ps._rates_base[ph77_degradation] = 0.253744;
     ps._rates_base[ph713_degradation] = 0.240119;
     ps._rates_base[ph1313_degradation] = 0.205776;
    cout << "no seg fault"<<endl;
    //setting up simulation
    CPUGPU_ALLOC(simulation_cuda, s, m, ps, 200, 50, 0.01);
    cout << "no seg fault"<<endl;
    s.initialize();
    cout << "no seg fault before sim"<<endl;
    //run simulation
    s.simulate_cuda(600);
    //s.print_delay();
    CPUGPU_DELETE(simulation_cuda, s);
    CPUGPU_DELETE(model, m);
    CPUGPU_DELETE(param_set, ps);

}
