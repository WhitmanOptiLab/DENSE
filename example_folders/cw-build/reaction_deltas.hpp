#include "utility/common_utils.hpp"
#include "core/reaction.hpp"
#include <cstddef>

STATIC_VAR int num_deltas_demand_slope = 1;
STATIC_VAR int deltas_demand_slope[] = {-1};
STATIC_VAR specie_id delta_ids_demand_slope[] = {P};

STATIC_VAR int num_deltas_demand_base = 1;
STATIC_VAR int deltas_demand_base[] = {1};
STATIC_VAR specie_id delta_ids_demand_base[] = {P};

STATIC_VAR int num_deltas_supply_slope = 1;
STATIC_VAR int deltas_supply_slope[] = {-1};
STATIC_VAR specie_id delta_ids_supply_slope[] = {P};

STATIC_VAR int num_deltas_supply_base = 1;
STATIC_VAR int deltas_supply_base[] = {-1};
STATIC_VAR specie_id delta_ids_supply_base[] = {P};