// In this file, we declare the X-Macro for the list of reactions to 
//   be simulated.  
// Each reaction must have a declared reaction rate function in 
//   model_impl.h
// For example, to declare three reactions named "one", "two", and "three", 
// use the syntax below
//#define LIST_OF_REACTIONS \
//  REACTION(one) \
//  REACTION(two) \
//  REACTION(three) \

#ifndef DELAY_REACTION
#define DELAY_REACTION REACTION
#define UNDO_DELAY_REACTION_DEF

/*
 
 */

  REACTION(ph1_synthesis)
  REACTION(ph1_degradation)
  REACTION(ph11_dissociation)
  REACTION(ph11_association)
  REACTION(pd_synthesis)
  REACTION(pd_degradation)
  REACTION(pma_synthesis)
  REACTION(pma_degradation)
  REACTION(pmb_synthesis)
  REACTION(pmb_degradation)
  REACTION(ph11_degradation)
  REACTION(pmaa_degradation)
  REACTION(pmab_degradation)
  REACTION(pmbb_degradation)
  REACTION(mh1_synthesis)
  REACTION(mh1_degradation)
  REACTION(md_synthesis)
  REACTION(md_degradation)
  REACTION(mma_synthesis)
  REACTION(mma_degradation)
  REACTION(mmb_synthesis)
  REACTION(mmb_degradation)
  REACTION(pmbb_association)
  REACTION(pmab_association)
  REACTION(pmaa_association)
  REACTION(pmaa_dissociation)
  REACTION(pmab_dissociation)
  REACTION(pmbb_dissociation)
#endif
#ifdef UNDO_DELAY_REACTION_DEF
#undef DELAY_REACTION 
#undef UNDO_DELAY_REACTION_DEF
#endif
/*
 REACTION(mespa_dissociation)
 REACTION(mespa_dissociation2)
 REACTION(mespa_dissociation3)
 REACTION(mespa_association)
 REACTION(mespa_association2)
 REACTION(mespa_association3)
REACTION(mespb_dissociation)
REACTION(mespb_dissociation2)
REACTION(mespb_dissociation3)
REACTION(mespb_association)
REACTION(mespb_association2)
REACTION(mespb_association3)*/

