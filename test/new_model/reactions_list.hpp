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
DELAY_REACTION(mh1_synthesis)
DELAY_REACTION(mh7_synthesis)
DELAY_REACTION(mh13_synthesis)
DELAY_REACTION(md_synthesis)
DELAY_REACTION(ph1_synthesis)
DELAY_REACTION(ph7_synthesis)
DELAY_REACTION(ph13_synthesis)
DELAY_REACTION(pd_synthesis)
REACTION(ph11_association)
REACTION(ph17_association)
REACTION(ph113_association)
REACTION(ph77_association)
REACTION(ph713_association)
REACTION(ph1313_association)
REACTION(ph11_dissociation)
REACTION(ph17_dissociation)
REACTION(ph113_dissociation)
REACTION(ph77_dissociation)
REACTION(ph713_dissociation)
REACTION(ph1313_dissociation)
#ifdef UNDO_DELAY_REACTION_DEF
#undef DELAY_REACTION 
#undef UNDO_DELAY_REACTION_DEF
#endif


