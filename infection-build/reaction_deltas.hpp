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

STATIC_VAR int num_deltas_influx = 1;
STATIC_VAR int deltas_influx[] = {1};
STATIC_VAR specie_id delta_ids_influx[] = {vulnerable};

STATIC_VAR int num_deltas_immunization = 2;
STATIC_VAR int deltas_immunization[] = {-1, 1};
STATIC_VAR specie_id delta_ids_immunization[] = {vulnerable, immune};

STATIC_VAR int num_deltas_infection = 2;
STATIC_VAR int deltas_infection[] = {-1, 1};
STATIC_VAR specie_id delta_ids_infection[] = {vulnerable, infected};

STATIC_VAR int num_deltas_recovery = 2;
STATIC_VAR int deltas_recovery[] = {-1, 1};
STATIC_VAR specie_id delta_ids_recovery[] = {infected, vulnerable};

STATIC_VAR int num_deltas_treatment = 2;
STATIC_VAR int deltas_treatment[] = {-1, 1};
STATIC_VAR specie_id delta_ids_treatment[] = {infected, immune};

STATIC_VAR int num_deltas_death = 2;
STATIC_VAR int deltas_death[] = {-1, 1};
STATIC_VAR specie_id delta_ids_death[] = {infected, dead};


