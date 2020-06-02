#ifndef DELAY_REACTION
#define DELAY_REACTION REACTION
#define UNDO_DELAY_REACTION_DEF
#endif
REACTION(make_lemon)
REACTION(make_sugar)
REACTION(make_lemonade)
REACTION(customer_purchase)
REACTION(buy_lemons)
REACTION(buy_sugar)
REACTION(make_quality)
REACTION(quality_to_satisfaction)
REACTION(satisfaction_to_tips)
#ifdef UNDO_DELAY_REACTION_DEF
#undef DELAY_REACTION
#undef UNDO_DELAY_REACTION_DEF
#endif
