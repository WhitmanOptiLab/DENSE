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

STATIC_VAR int num_deltas_red_warming = 2;
STATIC_VAR int deltas_red_warming[] = {19, -1};
STATIC_VAR specie_id delta_ids_red_warming[] = {red_box_T, red_room_dT};

STATIC_VAR int num_deltas_green_warming = 2;
STATIC_VAR int deltas_green_warming[] = {13, -2};
STATIC_VAR specie_id delta_ids_green_warming[] = {green_box_T, green_room_dT};

STATIC_VAR int num_deltas_blue_warming = 2;
STATIC_VAR int deltas_blue_warming[] = {7, -1};
STATIC_VAR specie_id delta_ids_blue_warming[] = {blue_box_T, blue_room_dT};

STATIC_VAR int num_deltas_red_green_diffusion = 2;
STATIC_VAR int deltas_red_green_diffusion[] = {3, -1};
STATIC_VAR specie_id delta_ids_red_green_diffusion[] = {green_room_dT, red_room_dT};

STATIC_VAR int num_deltas_green_blue_diffusion = 2;
STATIC_VAR int deltas_green_blue_diffusion[] = {1, -6};
STATIC_VAR specie_id delta_ids_green_blue_diffusion[] = {blue_room_dT, green_room_dT};