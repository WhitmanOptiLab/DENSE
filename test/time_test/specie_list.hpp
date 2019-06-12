// In this file, we declare the X-Macro for the list of reactions to
//   be simulated.
// Each reaction must have a declared reaction rate function in
//   model_impl.h
// For example, to declare three reactions named "one", "two", and "three",
// use the syntax below
//#define LIST_OF_SPECIES
//  SPECIE(specie_one)
//  SPECIE(two)
//  SPECIE(three)


#ifndef CRITICAL_SPECIE
#define CRITICAL_SPECIE SPECIE
#define UNDO_CRITICAL_SPECIE_DEF
#endif
SPECIE(ph1)
SPECIE(ph7)
SPECIE(ph13)
CRITICAL_SPECIE(pd)
SPECIE(mh1)
SPECIE(mh7)
SPECIE(mh13)
SPECIE(md)
CRITICAL_SPECIE(ph11)
SPECIE(ph17)
SPECIE(ph113)
SPECIE(ph77)
SPECIE(ph1313)
CRITICAL_SPECIE(ph713)
#ifdef UNDO_CRITICAL_SPECIE_DEF
#undef CRITICAL_SPECIE
#undef UNDO_CRITICAL_SPECIE_DEF
#endif
