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
    RATETYPE critical_values[5] = {691.147030,766.126001,1989.301517,1077.693514,2457.897298};
    RATETYPE delay_sets[8] = {7.012622,14.984612,8.023912,14.704954,1.243684,0.425859,0.409653,10.049570};
    RATETYPE rates_base[NUM_REACTIONS] = {30.012607,51.696917,38.245038,30.602979,
        
        0.499898,0.100750,0.244993,0.479945,
        
        54.038194,44.558513,23.450357,51.855638,
        
        0.119014,0.266534,0.445092,0.497927,
        
        0.025518,0.018589,0.026202,0.013235,
        
        0.151772,0.244074,0.223924,0.264807,
        
        0.224610,0.489840,0.486618,0.483102};
    
    for (int i = 0; i < 5; i++) {
        ps._critical_values[i] = critical_values[i];
    }
    
    for (int i = 0; i < 8; i++) {
        ps._delay_sets[i] = delay_sets[i];
    }
    for (int i = 0; i < NUM_REACTIONS; i++) {
        ps._rates_base[i] = rates_base[i];
    }
    
    //setting up simulation
    simulation s(m, ps, 200, 50,0.01);
    s._rates.update_rates(s._parameter_set._rates_base);
    s._delays.update_rates(s._parameter_set._delay_sets);
    s._critValues.update_rates(s._parameter_set._critical_values);
    s._cl.initialize(4,300,200);
    s._baby_cl.initialize();
    
    //run simulation
    s.execute();
}
