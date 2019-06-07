// In this file, we declare the X-Macro for the list of reactions to
//   be simulated.
// Each reaction must have a declared reaction rate function in
//   model_impl.h

#ifndef DELAY_REACTION
#define DELAY_REACTION REACTION
#define UNDO_DELAY_REACTION_DEF
#endif
REACTION(red_warming)
REACTION(green_warming)
REACTION(blue_warming)
REACTION(red_green_diffusion)
REACTION(green_blue_diffusion)
#ifdef UNDO_DELAY_REACTION_DEF
#undef DELAY_REACTION
#undef UNDO_DELAY_REACTION_DEF
#endif


