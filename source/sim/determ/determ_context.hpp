// A context defines a locale in which reactions take place and species
//   reside
#ifndef SIM_DETERM_DETERM_CONTEXT_HPP
#define SIM_DETERM_DETERM_CONTEXT_HPP

#include "utility/common_utils.hpp"
#include "determ.hpp"
#include "sim/cell_param.hpp"
#include <iostream>

IF_CUDA(__host__ __device__)
Real Deterministic_Simulation::Context::calculateNeighborAvg(specie_id sp, int delay) const{
    // Average the given cell's neighbors' concentrations
    Real sum=0;
    for (unsigned i = 0; i<_simulation._numNeighbors[_cell]; i++) {
        sum+=_simulation._baby_cl[sp][-delay][_simulation._neighbors[_cell][static_cast<unsigned>(i)]];
    }
    Real avg = sum/_simulation._numNeighbors[_cell];
    return avg;
}

IF_CUDA(__host__ __device__)
const Deterministic_Simulation::Context::SpecieRates Deterministic_Simulation::Context::calculateRatesOfChange(){
    //Step 1: for each reaction, compute reaction rate
    CUDA_Array<Real, NUM_REACTIONS> reaction_rates;
    #define REACTION(name) reaction_rates[name] = dense::model::reaction_##name.active_rate(*this);
        #include "reactions_list.hpp"
    #undef REACTION

    //Step 2: allocate specie concentration rate change array
    SpecieRates specie_deltas;
    for (int i = 0; i < NUM_SPECIES; i++)
      specie_deltas[i] = 0;

    //Step 3: for each reaction rate, for each specie it affects, accumulate its contributions
    #define REACTION(name) \
    const reaction<name>& r##name = dense::model::reaction_##name; \
    for (int j = 0; j < r##name.getNumDeltas(); j++) { \
        specie_deltas[delta_ids_##name[j]] += reaction_rates[name]*deltas_##name[j]; \
    }
    #include "reactions_list.hpp"
    #undef REACTION

    return specie_deltas;
}

IF_CUDA(__host__ __device__)
void Deterministic_Simulation::Context::updateCon(const Deterministic_Simulation::Context::SpecieRates& rates) {
    for (int i=0; i< NUM_SPECIES; i++){
        auto curr_rate = rates[i];
        _simulation._baby_cl[i][1][_cell]=_simulation._baby_cl[i][0][_cell]+ _simulation._step_size* curr_rate;
    }
}

#endif // CONTEXT_IMPL
