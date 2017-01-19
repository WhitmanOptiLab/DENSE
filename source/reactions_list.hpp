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

/*
 
 */

  REACTION(ph1_synthesis)
  REACTION(ph1_degradation)
  REACTION(ph1_dissociation)
  REACTION(ph1_association)
  REACTION(pd_synthesis)
  REACTION(pd_degradation)
  REACTION(mespa_synthesis)
  REACTION(mepsa_degradation)
  REACTION(mespb_synthesis)
  REACTION(mepsb_degradation)
  REACTION(ph11_degradation)
  REACTION(pm11_degradation)
  REACTION(pm12_degradation)
  REACTION(pm22_degradation)
  REACTION(mh1_synthesis)
  REACTION(mh1_degradation)
  REACTION(md_synthesis)
  REACTION(md_degradation)
  REACTION(mm1_synthesis)
  REACTION(mm1_degradation)
  REACTION(mm2_synthesis)
  REACTION(mm2_degradation)
  REACTION(pm22_association)
  REACTION(pm12_association)
  REACTION(pm11_association)
  REACTION(pm11_dessociation)
  REACTION(pm12_dessociation)
  REACTION(pm22_dessociation)
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

