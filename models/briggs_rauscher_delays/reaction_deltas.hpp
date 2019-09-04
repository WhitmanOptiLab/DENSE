#include "utility/common_utils.hpp"
#include "core/reaction.hpp"

#include <cstddef>

/*

Define each reaction's reactants and products in reaction_deltas.hpp.
Say a reaction enumerated as R_ONE has the following chemical formula:

                           2A + B --> C

The proper way to define that reaction's state change vector is as follows:

STATIC_VAR int num_deltas_R_ONE = 3;
STATIC_VAR int deltas_R_ONE[] = {-2, -1, 1};
STATIC_VAR specie_id delta_ids_R_ONE[] = {A, B, C};

*/

//Reaction that synthesizes single HIO molecule
STATIC_VAR int num_deltas_radical = 1;
STATIC_VAR int deltas_radical[] = {1};
STATIC_VAR specie_id delta_ids_radical[] = {HIO};

//Non-radical reaction(takes 2 HIO molecules to create 1 I molucule)
STATIC_VAR int num_deltas_non_radical = 2;
STATIC_VAR int deltas_non_radical[] = {-2, 1};
STATIC_VAR specie_id delta_ids_non_radical[] = {HIO, I};

//Reaction that gets rid of single HIO molecule
STATIC_VAR int num_deltas_hio_decay = 1;
STATIC_VAR int deltas_hio_decay[] = {-1};
STATIC_VAR specie_id delta_ids_hio_decay[] = {HIO};

//Reaction that gets rid of single I molecule
STATIC_VAR int num_deltas_i_decay = 1;
STATIC_VAR int deltas_i_decay[] = {-1};
STATIC_VAR specie_id delta_ids_i_decay[] = {I};

STATIC_VAR int num_deltas_i_decay_delay = 1;
STATIC_VAR int deltas_i_decay_delay[] = {-1};
STATIC_VAR specie_id delta_ids_i_decay_delay[] = {I};

STATIC_VAR int num_deltas_hio_decay_delay = 1;
STATIC_VAR int deltas_hio_decay_delay[] = {-1};
STATIC_VAR specie_id delta_ids_hio_decay_delay[] = {HIO};
