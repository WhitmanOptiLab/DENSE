#include "utility/common_utils.hpp"
#include "core/reaction.hpp"

#include <cstddef>

STATIC_VAR int num_deltas_make_lemonade = 3;
STATIC_VAR int deltas_make_lemonade[] = {-2,-3,1};
STATIC_VAR specie_id delta_ids_make_lemonade[] = {lemon,sugar,lemonade};

STATIC_VAR int num_deltas_make_lemon = 1;
STATIC_VAR int deltas_make_lemon[] = {1};
STATIC_VAR specie_id delta_ids_make_lemon[] = {lemon};

STATIC_VAR int num_deltas_make_sugar = 1;
STATIC_VAR int deltas_make_sugar[] = {1};
STATIC_VAR specie_id delta_ids_make_sugar[] = {sugar};

STATIC_VAR int num_deltas_customer_purchase = 2;
STATIC_VAR int deltas_customer_purchase[] = {-1,2};
STATIC_VAR specie_id delta_ids_customer_purchase[] = {lemonade,money};

STATIC_VAR int num_deltas_buy_lemons = 2;
STATIC_VAR int deltas_buy_lemons[] = {-3,2};
STATIC_VAR specie_id delta_ids_buy_lemons[] = {money,lemon};

STATIC_VAR int num_deltas_buy_sugar = 2;
STATIC_VAR int deltas_buy_sugar[] = {-2,25};
STATIC_VAR specie_id delta_ids_buy_sugar[] = {money,sugar};

STATIC_VAR int num_deltas_make_quality = 2;
STATIC_VAR int deltas_make_quality[] = {-4,1};
STATIC_VAR specie_id delta_ids_make_quality[] = {lemonade, good_lemonade};

STATIC_VAR int num_deltas_quality_to_satisfaction = 3;
STATIC_VAR int deltas_quality_to_satisfaction[] = {-2,-1,1};
STATIC_VAR specie_id delta_ids_quality_to_satisfaction[] = {good_lemonade, lemonade, customer_satisfaction};

STATIC_VAR int num_deltas_satisfaction_to_tips = 2;
STATIC_VAR int deltas_satisfaction_to_tips[] = {-1,4};
STATIC_VAR specie_id delta_ids_satisfaction_to_tips[] = {customer_satisfaction, money};
