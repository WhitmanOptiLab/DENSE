// In this file, we declare the X-Macro for the list of reactions to
//   be simulated.
// Each reaction must have a declared reaction rate function in
//   model_impl.h

#ifndef CRITICAL_SPECIE
#define CRITICAL_SPECIE SPECIE
#define UNDO_CRITICAL_SPECIE_DEF
#endif
CRITICAL_SPECIE(red_room_dT)
SPECIE(red_box_T)
CRITICAL_SPECIE(green_room_dT)
SPECIE(green_box_T)
CRITICAL_SPECIE(blue_room_dT)
SPECIE(blue_box_T)
#ifdef UNDO_CRITICAL_SPECIE_DEF
#undef CRITICAL_SPECIE
#undef UNDO_CRITICAL_SPECIE_DEF
#endif
