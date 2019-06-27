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
STATIC_VAR int num_deltas_activator_diffusion_rate = 2;
STATIC_VAR int deltas_activator_diffusion_rate[] = {-1,-2};
STATIC_VAR specie_id delta_ids_activator_diffusion_rate[] = {activator,activator_dt};

STATIC_VAR int num_deltas_inhibitor_diffusion_rate = 2;
STATIC_VAR int deltas_inhibitor_diffusion_rate[] = {-2,-1};
STATIC_VAR specie_id delta_ids_inhibitor_diffusion_rate[] = {inhibitor,inhibitor_dt};

STATIC_VAR int num_deltas_activator_dCon = 2;
STATIC_VAR int deltas_activator_dCon[] = {-1,1};
STATIC_VAR specie_id delta_ids_activator_dCon[] = {activator,activator_dt};

STATIC_VAR int num_deltas_inhibitor_dCon = 1;
STATIC_VAR int deltas_inhibitor_dCon[] = {-1};
STATIC_VAR specie_id delta_ids_inhibitor_dCon[] = {inhibitor};

STATIC_VAR int num_deltas_activator_synthesis = 3;
STATIC_VAR int deltas_activator_synthesis[] = {-1,1,1};
STATIC_VAR specie_id delta_ids_activator_synthesis[] = {activator_dt,activator,inhibitor_dt};

STATIC_VAR int num_deltas_inhibitor_synthesis = 1;
STATIC_VAR int deltas_inhibitor_synthesis[] = {1};
STATIC_VAR specie_id delta_ids_inhibitor_synthesis[] = {inhibitor};
