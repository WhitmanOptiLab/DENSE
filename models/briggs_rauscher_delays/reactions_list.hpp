// In this file, we declare the X-Macro for the list of reactions to 
//   be simulated.  
// Each reaction must have a declared reaction rate function in 
//   model_impl.h
// For example, to declare three reactions named "one", "two", and "three", 
// of which "three" is a delayed reaction, use the syntax below.
//
//  REACTION(one)
//  REACTION(two)
//  DELAY_REACTION(three)

#ifndef DELAY_REACTION
#define DELAY_REACTION REACTION
#define UNDO_DELAY_REACTION_DEF
#endif

//Briggs-Rauscher radical reaction to synthesize HIO and initialize concentrations
REACTION(radical) 
////Briggs-Rauscher radical reaction to convert HIO to I 
REACTION(non_radical)
//Reaction to balance out excess HIO created
REACTION(hio_decay)
//Reaction to balance out excess I created	
REACTION(i_decay)

DELAY_REACTION(i_decay_delay)

DELAY_REACTION(hio_decay_delay)

#ifdef UNDO_DELAY_REACTION_DEF
#undef DELAY_REACTION 
#undef UNDO_DELAY_REACTION_DEF
#endif
