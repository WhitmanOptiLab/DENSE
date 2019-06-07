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

/* ./simulation -p ./param_sets.csv -c 10 -w 10 -t 20  -u 0.5 -e "output.csv"
*/
STATIC_VAR int num_deltas_fe2_reaction = 3;
STATIC_VAR int deltas_fe2_reaction[] = {-2,-2, 1};
STATIC_VAR specie_id delta_ids_fe2_reaction[] = {BR,FE2,FE3};

STATIC_VAR int num_deltas_fe3_reaction = 3;
STATIC_VAR int deltas_fe3_reaction[] = {-3, 4,4};
STATIC_VAR specie_id delta_ids_fe3_reaction[] = {FE3,FE2,BR};

STATIC_VAR int num_deltas_br_synthesis = 1;
STATIC_VAR int deltas_br_synthesis[] = {1};
STATIC_VAR specie_id delta_ids_br_synthesis[] = {BR};


STATIC_VAR int num_deltas_fe2_synthesis = 1;
STATIC_VAR int deltas_fe2_synthesis[] = {2};
STATIC_VAR specie_id delta_ids_fe2_synthesis[] = {FE2};

