#include "simulation.hpp"
#include <iostream>

int main() {
    model m;
    m._using_perturb = false;
    m._using_gradients = false;
    for (int i = 0; i < NUM_SPECIES; i++) {
        m._has_gradient[i] = false;
    }
    
    param_set ps;
    RATETYPE delay_sets = {7.012622,14.984612,8.023912,14.704954,1.243684,0.425859,0.409653,10.049570};
    RATETYPE rates_base = {30.012607,51.696917,38.245038,30.602979,
        
        0.499898,0.100750,0.244993,0.479945,
        
        54.038194,44.558513,23.450357,51.855638,
        
        0.119014,0.266534,0.445092,0.497927,
        
        0.025518,0.018589,0.026202,0.013235,
        
        0.151772,0.244074,0.223924,0.264807,
        
        0.224610,0.489840,0.486618,0.483102};
    
    for (int i = 0; i < NUM_DELAY_REACTIONS; i++) {
        m._delay_sets[i] = delay_sets[i];
    }
    
    for (int i = 0; i < NUM_REACTIONS; i++) {
        m._rates_base[i] = rates_base[i];
    }
    
    simulation s(m, ps);
    s.test_sim();
}
