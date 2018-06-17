// A context defines a locale in which reactions take place and species
//   reside
#ifndef SIM_STOCH_STOCH_CONTEXT_HPP
#define SIM_STOCH_STOCH_CONTEXT_HPP

#include "stoch.hpp"
#include "sim/cell_param.hpp"
#include "core/reaction.hpp"
#include <iostream>

/*
 * CALCULATENEIGHBORAVG
 * arg "sp": the specie to average from the surrounding cells
 * arg "delay": unused, but used in deterministic context. Kept for polymorphism
 * returns "avg": average concentration of specie in current and neighboring cells
*/
IF_CUDA(__host__ __device__)
Real simulation_stoch::ContextStoch::calculateNeighborAvg(specie_id sp, int delay) const{
    Real sum=0;
    for (int i=0; i<_simulation._numNeighbors[_cell]; i++){
        sum+=_simulation.concs[_simulation._neighbors[_cell][i]][sp];
    }
    Real avg = sum/_simulation._numNeighbors[_cell];
    return avg;
}

/*
 * UPDATEPROPENSITIES
 * recalculates the propensities of reactions affected by the firing of "rid"
 * arg "rid": the reaction that fired
*/
IF_CUDA(__host__ __device__)
void simulation_stoch::ContextStoch::updatePropensities(reaction_id rid){
    const model& _model = _simulation._model;

    #define REACTION(name) \
    for (std::size_t i=0; i<_simulation.propensity_network[rid].size(); i++) { \
        if ( name == _simulation.propensity_network[rid][i] ) { \
            _simulation.propensities[_cell][name] = _model.reaction_##name.active_rate(*this); \
        } \
    } \
\
    for (std::size_t r=0; r<_simulation.neighbor_propensity_network[rid].size(); r++) { \
        if (name == _simulation.neighbor_propensity_network[rid][r]) { \
            for (int n=0; n<_simulation._numNeighbors[_cell]; n++) { \
                int n_cell = _simulation._neighbors[_cell][n]; \
                ContextStoch neighbor(_simulation,n_cell); \
                _simulation.propensities[n_cell][name] = _model.reaction_##name.active_rate(neighbor); \
            } \
        } \
    }
    #include "reactions_list.hpp"
    #undef REACTION
}

/*
 * GETTOTALPROPENSITY
 * sums the propensities of every reaction in every cell
 * called by "generateTau" in simulation_stoch.cpp
 * return "sum": the propensity sum
*/
IF_CUDA(__host__ __device__)
Real simulation_stoch::ContextStoch::getTotalPropensity(){
    Real sum = 0;
    for (int c=0; c<_simulation._cells_total; c++){
      for (int r=0; r<NUM_REACTIONS; r++){
        sum+=_simulation.propensities[c][r];
      }
    }
    return sum;
}

/*
 * CHOOSEREACTION
 * randomly chooses a reaction biased by their propensities
 * arg "propensity_portion": the propensity sum times a random variable between 0.0 and 1.0
 * return "j": the index of the reaction chosen.
*/
IF_CUDA(__host__ __device__)
int simulation_stoch::ContextStoch::chooseReaction(Real propensity_portion) {
  Real sum = 0;
  int c, s;

  for (c = 0; c < _simulation._cells_total; c++) {
    for (s = 0; s < NUM_REACTIONS; s++) {
      sum += _simulation.propensities[c][s];

      if (sum > propensity_portion) {
        int j = (c * NUM_REACTIONS) + s;
        return j;
      }
    }
  }

  int j = ((c - 1) * NUM_REACTIONS) + (s - 1);
  return j;
}


#endif // CONTEXT_STOCH
