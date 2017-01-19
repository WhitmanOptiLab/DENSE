// A context defines a locale in which reactions take place and species 
//   reside

#include "simulation.hpp"
#include "rates.hpp"
#include "contexts.hpp"
#include "model_impl.hpp"
#include <iostream>


using namespace std;

void context::calculateNeighbourAvg(specie sp, double[] old_cells_mrna, concentration_level& cl){
    int neighbors[NUM_DD_INDICES][NEIGHBORS_2D];
    for (int j = 0; j < NUM_DD_INDICES; j++) {
        // 2D neighbors are precalculated and simply copied from the structure as needed
        int cell = old_cells_mrna[IMH1 + j];
        memcpy(neighbors[IMH1 + j], _simulation.neighbors[cell], sizeof(int) * NEIGHBORS_2D);
    }
    
    // For each mRNA concentration, average the given cell's neighbors' Delta protein concentrations
    for (int j = 0; j < NUM_DD_INDICES; j++) {
        int* cells = neighbors[IMH1 + j];
        int cell = old_cells_mrna[IMH1 + j];
        int time = WRAP(stc.time_cur - delays[j], _simulation.max_delay_size);
        concentration_level<double>::cell cur_cons = cl[CPDELTA][time];
        double sum=0;
        if (cell % _simulation.width_total == cl.active_start_record[time]) {
            sum = (cur_cons[cells[0]] + cur_cons[cells[3]] + cur_cons[cells[4]] + cur_cons[cells[5]]) / 4;
        } else if (cell % _simulation.width_total == cl.active_start_record[time]) {
            sum = (cur_cons[cells[0]] + cur_cons[cells[1]] + cur_cons[cells[2]] + cur_cons[cells[3]]) / 4;
        } else {
            sum = (cur_cons[cells[0]] + cur_cons[cells[1]] + cur_cons[cells[2]] + cur_cons[cells[3]] + cur_cons[cells[4]] + cur_cons[cells[5]]) / 6;
        }
        avg_delays[IMH1 + j] = sum;
    }
}

double[] calculateRates(){
    
}

void context::updateCon(concentration_level& cl, double[] rates){
    //double step_size= _simulation.step_size;
    int j= _simulation._j;
    double curr_rate=0;
    for (int i=0; i< NUM_SPECIES; i++){
        curr_rate= rates[i];
        cl[i][j+1][_cell]=cl[i][j][cell]+ _simulation.step_size* curr_rate;
    }
    
}
