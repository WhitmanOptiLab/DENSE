/*

Declare species in `specie_list.hpp`. List the specie names between the
  two sets of C++ macros (the lines that begin with `#`) in the same format
  as below. The following example lists two species, `alpha` and `bravo`,
  and one critical speice, `charlie`.

    SPECIE(alpha)
    SPECIE(bravo)
    CRITICAL_SPECIE(charlie)

*/
#ifndef CRITICAL_SPECIE    
#define CRITICAL_SPECIE SPECIE
#define UNDO_CRITICAL_SPECIE_DEF
#endif

//DEFINE SPECIES HERE

#ifdef UNDO_CRITICAL_SPECIE_DEF
#undef CRITICAL_SPECIE
#undef UNDO_CRITICAL_SPECIE_DEF
#endif
