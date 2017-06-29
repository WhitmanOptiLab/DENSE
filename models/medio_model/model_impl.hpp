// In this header file, define your model!
// This includes functions to describe each reaction.
// Make sure that you've first completed reaction_list.h and specie_list.h
#ifndef MODEL_IMPL_H
#define MODEL_IMPL_H
#include "reaction.hpp"
#include "specie.hpp"
#include "model.hpp"
#include "simulation_base.hpp"
//#include "context.hpp"
#include <cstddef>
#include <algorithm>


// Next, define all of your reaction rate functions
// For example, if you enumerated a reaction R_ONE, you should declare a 
//   function like this:
//
// RATETYPE reaction<R_ONE>::active_rate(const Ctxt c) const { return 6.0; }
// 
// Or, for a more interesting reaction rate, you might do something like
// 
// RATETYPE reaction<R_TWO>::active_rate(const Ctxt c) const {
//   return c.rate[R_TWO] * c.concentration[SPECIE_ONE] * 
//                                   c.neighbors.concentration[SPECIE_TWO];
// }
template<>
template<class Ctxt>
RATETYPE reaction<mRNA_synthesis>::active_rate(const Ctxt& c) const {
    return c.getRate(mRNA_synthesis) * std::max(c.getCritVal(rcrit_protein)-c.getCon(protein), RATETYPE(0.0));
}

template<>
template<class Ctxt>
RATETYPE reaction<protein_synthesis>::active_rate(const Ctxt& c) const {
    return c.getRate(protein_synthesis) * c.getCon(mRNA, c.getDelay(dreact_protein_synthesis));
}


#endif // MODEL_IMPL_H
