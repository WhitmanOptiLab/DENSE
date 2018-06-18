/*
Declare reactions in `reactions_list.hpp`. List the reaction names between the two sets of C++ macros (the lines that begin with `#`) in the same format as below. The following example lists one delay reaction, `alpha_synthesis`, and three normal reactions, `bravo_synthesis`, `alpha_degredation`, and `bravo_degredation`. While this particular reaction naming scheme is not required, it can be helpful.

    DELAY_REACTION(alpha_synthesis)
    REACTION(bravo_synthesis)
    REACTION(alpha_degredation)
    REACTION(bravo_degredation)
*/

#ifndef DELAY_REACTION
#define DELAY_REACTION REACTION
#define UNDO_DELAY_REACTION_DEF
#endif

//DEFINE REACTIONS HERE

#ifdef UNDO_DELAY_REACTION_DEF
#undef DELAY_REACTION 
#undef UNDO_DELAY_REACTION_DEF
#endif


