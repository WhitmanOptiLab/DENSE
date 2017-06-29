// A context defines a locale in which reactions take place and species 
//   reside
#ifndef CONTEXT_STOCH
#define CONTEXT_STOCH

#include "simulation_stoch.hpp"
#include "cell_param.hpp"
#include "reaction.hpp"
//#include "context.hpp"
#include <iostream>
using namespace std;

CPUGPU_FUNC
RATETYPE simulation_stoch::ContextStoch::calculateNeighborAvg(specie_id sp, int delay) const{
    /*
    //int NEIGHBORS_2D= _simulation.NEIGHBORS_2D;
    //int neighbors[NUM_DELAY_REACTIONS][NEIGHBORS_2D];
    CPUGPU_TempArray<int, 6>& cells = _simulation._neighbors[_cell];

    //memcpy(neighbors[sp.index], _simulation.neighbors[_cell], sizeof(int) * NEIGHBORS_2D);
    //delay = rs[sp][_cell] / _simulation._step_size;
    // For each mRNA concentration, average the given cell's neighbors' Delta protein concentrations
    //int* cells = _simulation._neighbors[_cell];
    //int time = WRAP(_simulation._j - delay, _simulation._delay_size[sp.index]);
    // TODO: remove CPDELTA hardcoding
    baby_cl::cell cur_cons = _simulation.concs[pd][-delay];
    RATETYPE sum=0;
    //since the tissue is not growing now
    //start is 0 and end is 10, instead of_simulation.active_start_record[time] and_simulation.active_end_record[time]
    if (_cell % _simulation._width_total == 0) {
        sum = (cur_cons[cells[0]] + cur_cons[cells[3]] + cur_cons[cells[4]] + cur_cons[cells[5]]) / 4;
    } else if (_cell % _simulation._width_total == 10) {
        sum = (cur_cons[cells[0]] + cur_cons[cells[1]] + cur_cons[cells[2]] + cur_cons[cells[3]]) / 4;
    } else {
        sum = (cur_cons[cells[0]] + cur_cons[cells[1]] + cur_cons[cells[2]] + cur_cons[cells[3]] + cur_cons[cells[4]] + cur_cons[cells[5]]) / 6;
    }
    return sum;
    */
    RATETYPE sum=0;

    for (int i=0; i<_simulation._neighbors[_cell].size(); i++){
        sum+=_simulation.concs[_simulation._neighbors[_cell][i]][sp];
    }
    
    return sum/_simulation._neighbors[_cell].size();
}

CPUGPU_FUNC
void simulation_stoch::ContextStoch::updatePropensities(reaction_id rid){
    const model& _model = _simulation._model;

    #define REACTION(name) \
    for (int i=0; i<_simulation.propensity_network[rid].size(); i++) { \
        if ( name == _simulation.propensity_network[rid][i] ) { \
            _simulation.propensities[_cell][name] = _model.reaction_##name.active_rate(*this); \
        } \
    } \
\
    for (int r=0; r<_simulation.neighbor_propensity_network[rid].size(); r++) { \
        if (name == _simulation.neighbor_propensity_network[rid][r]) { \
            for (int n=0; n<_simulation._neighbors[_cell].size(); n++) { \
                int n_cell = _simulation._neighbors[_cell][n]; \
                ContextStoch neighbor(_simulation,n_cell); \
                _simulation.propensities[n_cell][name] = _model.reaction_##name.active_rate(neighbor); \
            } \
        } \
    }
    #include "reactions_list.hpp"
    #undef REACTION
}

CPUGPU_FUNC
RATETYPE simulation_stoch::ContextStoch::getTotalPropensity(){
    RATETYPE sum = 0;
    for (int c=0; c<_simulation._cells_total; c++){
      for (int r=0; r<NUM_REACTIONS; r++){
        sum+=_simulation.propensities[c][r];
      }
    }
    return sum;
}

CPUGPU_FUNC
int simulation_stoch::ContextStoch::chooseReaction(RATETYPE propensity_portion){
    RATETYPE sum=0;
    int c,s;

//    _simulation.chooseReactionCount++;

    for (c=0; c<_simulation._cells_total; c++){
      for (s=0; s<NUM_REACTIONS; s++){
        sum+=_simulation.propensities[c][s];
//        if (_simulation.chooseReactionCount%1000000 == 0) {
//            cout << "    propen of react " << s << " = " << _simulation.propensities[c][s] << endl;
//        }
	    if (sum>propensity_portion){
            return (c*NUM_REACTIONS)+s;
	    }
      }
    }
    return ((c-1)*NUM_REACTIONS)+(s-1);
}


#endif // CONTEXT_STOCH
