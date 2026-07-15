#ifndef DELAY_REACTION
#define DELAY_REACTION REACTION
#define UNDO_DELAY_REACTION_DEF
#endif

// dp/dt = speed*a*p + speed*b - speed*c*p_delayed - speed*d = speed*(demand - supply)

REACTION(demand_slope)          // speed*a*p           (the a*p_now part of demand)
REACTION(demand_base)           // speed*b              (the +b part of demand)
DELAY_REACTION(supply_slope)    // -speed*c*p_delayed   (the c*p_delayed part of supply); carries tau
REACTION(supply_base)           // -speed*d              (the +d part of supply)

#ifdef UNDO_DELAY_REACTION_DEF
#undef DELAY_REACTION
#undef UNDO_DELAY_REACTION_DEF
#endif