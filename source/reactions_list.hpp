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

  DELAY_REACTION(ph1_synthesis)
  REACTION(ph1_degradation)
  REACTION(ph11_dissociation)
  REACTION(ph11_association)
  DELAY_REACTION(pd_synthesis)
  REACTION(pd_degradation)
  DELAY_REACTION(pm1_synthesis)
  REACTION(pm1_degradation)
  DELAY_REACTION(pm2_synthesis)
  REACTION(pm2_degradation)
  REACTION(ph11_degradation)
  REACTION(pm11_degradation)
  REACTION(pm12_degradation)
  REACTION(pm22_degradation)
  DELAY_REACTION(mh1_synthesis)
  REACTION(mh1_degradation)
  DELAY_REACTION(md_synthesis)
  REACTION(md_degradation)
  DELAY_REACTION(mm1_synthesis)
  REACTION(mm1_degradation)
  DELAY_REACTION(mm2_synthesis)
  REACTION(mm2_degradation)
  REACTION(pm22_association)
  REACTION(pm12_association)
  REACTION(pm11_association)
  REACTION(pm11_dissociation)
  REACTION(pm12_dissociation)
  REACTION(pm22_dissociation)
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

