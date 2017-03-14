// In this file, we declare the X-Macro for the list of reactions to 
//   be simulated.  
// Each reaction must have a declared reaction rate function in 
//   model_impl.h
// For example, to declare three reactions named "one", "two", and "three", 
// use the syntax below
//#define LIST_OF_SPECIES \
//  SPECIE(specie_one) \
//  SPECIE(two) \
//  SPECIE(three) \


#ifndef CRITICAL_SPECIE
#define CRITICAL_SPECIE SPECIE
#define UNDO_CRITICAL_SPECIE_DEF
#endif

SPECIE(ph1)
CRITICAL_SPECIE(pd) 
CRITICAL_SPECIE(pm1) 
CRITICAL_SPECIE(pm2) 
CRITICAL_SPECIE(ph11) 
SPECIE(pm11) 
SPECIE(pm12) 
CRITICAL_SPECIE(pm22) 
SPECIE(mh1) 
SPECIE(md) 
SPECIE(mm1) 
SPECIE(mm2)


#ifdef UNDO_CRITICAL_SPECIE_DEF
#undef CRITICAL_SPECIE
#undef UNDO_CRITICAL_SPECIE_DEF
#endif

