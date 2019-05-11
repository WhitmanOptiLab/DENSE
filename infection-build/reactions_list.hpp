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
REACTION(influx)
REACTION(immunization)
REACTION(infection)
REACTION(recovery)
REACTION(treatment)
REACTION(death)
#ifdef UNDO_DELAY_REACTION_DEF
#undef DELAY_REACTION
#undef UNDO_DELAY_REACTION_DEF
#endif


