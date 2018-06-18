#include "core/reaction.hpp"

#include <cstddef>

/*

Define each reaction's reactants and products in `reaction_deltas.hpp`.
Say a reaction enumerated as `R_ONE` has the following chemical formula:

                           2A + B --> C

The proper way to define that reaction's state change vector is as follows:
        
        STATIC_VAR int num_deltas_R_ONE = 3;
        STATIC_VAR int deltas_R_ONE[] = {-2, -1, 1};
        STATIC_VAR specie_id delta_ids_R_ONE[] = {A, B, C};

*/



