#ifndef CRITICAL_SPECIE
#define CRITICAL_SPECIE SPECIE
#define UNDO_CRITICAL_SPECIE_DEF
#endif

SPECIE(P)

// Parameter-only critical species -- raw CWParams fields (tau excluded,
// it's handled as a delay parameter on p_3 instead). Order matches
// your struct: a,b,c,d,speed
CRITICAL_SPECIE(a)
CRITICAL_SPECIE(b)
CRITICAL_SPECIE(c)
CRITICAL_SPECIE(d)
CRITICAL_SPECIE(speed)

#ifdef UNDO_CRITICAL_SPECIE_DEF
#undef CRITICAL_SPECIE
#undef UNDO_CRITICAL_SPECIE_DEF
#endif