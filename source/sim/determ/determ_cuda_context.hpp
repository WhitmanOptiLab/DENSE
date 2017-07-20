// A context defines a locale in which reactions take place and species 
//   reside
#ifndef SIM_DETERM_DETERM_CUDA_CONTEXT
#define SIM_DETERM_DETERM_CUDA_CONTEXT

#include "determ_cuda.hpp"
#include "sim/cell_param.hpp"
#include <iostream>
using namespace std;

CPUGPU_FUNC
RATETYPE simulation_cuda::Context::calculateNeighborAvg(specie_id sp, int delay) const{
    // Average the given cell's neighbors' concentrations
    RATETYPE sum=0;
    for (int i=0; i<_simulation._numNeighbors[_cell]; i++){
        sum+=_simulation._baby_cl_cuda[sp][-delay][_simulation._neighbors[_cell][i]];
    }
    RATETYPE avg = sum/_simulation._numNeighbors[_cell]; 
    return avg;
}

CPUGPU_FUNC
const simulation_cuda::Context::SpecieRates simulation_cuda::Context::calculateRatesOfChange(){
    const model& _model = _simulation._model;
    
    //Step 1: for each reaction, compute reaction rate
    CPUGPU_TempArray<RATETYPE, NUM_REACTIONS> reaction_rates;
    for (int i = 0; i < NUM_REACTIONS; i++) { reaction_rates[i] = 1.0f; }
    #define REACTION(name) reaction_rates[name] = _model.reaction_##name.active_rate(*this);
        #include "reactions_list.hpp"
    #undef REACTION
    
    //Step 2: allocate specie concentration rate change array
    SpecieRates specie_deltas;
    for (int i = 0; i < NUM_SPECIES; i++) 
      specie_deltas[i] = 0;
    
    //Step 3: for each reaction rate, for each specie it affects, accumulate its contributions
    #define REACTION(name) \
    const reaction<name>& r##name = _model.reaction_##name; \
    for (int j = 0; j < r##name.getNumDeltas(); j++) { \
        specie_deltas[delta_ids_##name[j]] += reaction_rates[name]*deltas_##name[j]; \
    }
    #include "reactions_list.hpp"
    #undef REACTION
    
    return specie_deltas;
}

CPUGPU_FUNC
void simulation_cuda::Context::updateCon(const simulation_cuda::Context::SpecieRates& rates){
    //double step_size= _simulation.step_size;
    
    double curr_rate=0;
    for (int i=0; i< NUM_SPECIES; i++){
        curr_rate= rates[i];
        static_cast<simulation_cuda&>(_simulation)._baby_cl_cuda[i][1][_cell]=
          static_cast<simulation_cuda&>(_simulation)._baby_cl_cuda[i][0][_cell] + _simulation._step_size* curr_rate;
    }
    
}

#endif // CONTEXT_IMPL
