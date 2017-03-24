#include "simulation.hpp"
#include <iostream>

int main() {
    
    //setting up model
    model m;
    m._using_perturb = false;
    m._using_gradients = false;
    for (int i = 0; i < NUM_SPECIES; i++) {
        m._has_gradient[i] = false;
    }
    
    //setting up param_set
    param_set ps;
    RATETYPE critical_values[NUM_CRITICAL_SPECIES] = {713.625,740.349,201.173};
    RATETYPE delay_sets[NUM_DELAY_REACTIONS] = {10.6427,9.15516,0,11.2572,1.72161,1.93494,0.805212,11.3695};
    RATETYPE rates_base[NUM_REACTIONS] = {
        55.282,55.8188,44.856,35.3622,
        0.215387,0.37152,0.10447,0.453508,
        25.8443,40.6742,35.8334,54.2915,
        0.210157,0.269844,0.335617,0.282566,
        0.020873,0.002822,0.027891,0.027807,0.017196,0.020549,
        0.280481,0.256498,0.030745,0.296646,0.067092,0.195206,
        0.296753,0.324761,0.144681,0.253744,0.240119,0.205776};
    
    for (int i = 0; i < 5; i++) {
        ps._critical_values[i] = critical_values[i];
    }
    
    for (int i = 0; i < 8; i++) {
        ps._delay_sets[i] = delay_sets[i];
    }
    for (int i = 0; i < NUM_REACTIONS; i++) {
        ps._rates_base[i] = rates_base[i];
    }
    cout << "no seg fault"<<endl;
    //setting up simulation
    simulation s(m, ps, 200, 50,0.01);
    cout << "no seg fault"<<endl;
    s.initialize();
    cout << "no seg fault"<<endl;
    //run simulation
    s.simulate(.5);
    //s.print_delay();
}
