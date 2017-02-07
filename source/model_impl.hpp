// In this header file, define your model!
// This includes functions to describe each reaction.
// Make sure that you've first completed reaction_list.h and specie_list.h
#ifndef MODEL_IMPL_H
#define MODEL_IMPL_H
#include "reaction.hpp"
#include "specie.hpp"
#include "model.hpp"

#include <cstddef>

// Next, define all of your reaction rate functions
// For example, if you enumerated a reaction R_ONE, you should declare a 
//   function like this:
//
// double reaction<R_ONE>::active_rate(const Context<double> c) const { return 6.0; }
// 
// Or, for a more interesting reaction rate, you might do something like
// 
// double reaction<R_TWO>::active_rate(const Context<double> c) const {
//   return c.rate[R_TWO] * c.concentration[SPECIE_ONE] * 
//                                   c.neighbors.concentration[SPECIE_TWO];
// }
/*
template<>
double reaction<one>::active_rate(const Context<double>& c) const {
  return 6.0;
}

template<>
double reaction<two>::active_rate(const Context<double>& c) const {
  return 3.0;
}

template<>
double reaction<three>::active_rate(const Context<double>& c) const {
  return 8.0;
}
*/

template<>
double reaction<ph1_synthesis>::active_rate(const Context<double>& c) const {
    return 6.0;
}

template<>
double reaction<ph1_degradation>::active_rate(const Context<double>& c) const {
    return 6.0;
}

template<>
double reaction<ph1_dissociation>::active_rate(const Context<double>& c) const {
    return 6.0;
}

template<>
double reaction<ph1_association>::active_rate(const Context<double>& c) const {
    return 6.0;
}

template<>
double reaction<pd_synthesis>::active_rate(const Context<double>& c) const {
    return 6.0;
}

template<>
double reaction<pd_degradation>::active_rate(const Context<double>& c) const {
    return 6.0;
}

template<>
double reaction<pma_synthesis>::active_rate(const Context<double>& c) const {
    return 6.0;
}

template<>
double reaction<pma_degradation>::active_rate(const Context<double>& c) const {
    return 6.0;
}

template<>
double reaction<pmbb_association>::active_rate(const Context<double>& c) const {
    return 6.0;
}

template<>
double reaction<pmaa_association>::active_rate(const Context<double>& c) const {
    return 6.0;
}

template<>
double reaction<pmab_association>::active_rate(const Context<double>& c) const {
    return 6.0;
}

template<>
double reaction<pmaa_dissociation>::active_rate(const Context<double>& c) const {
    return 6.0;
}

template<>
double reaction<pmab_dissociation>::active_rate(const Context<double>& c) const {
    return 6.0;
}

template<>
double reaction<pmbb_dissociation>::active_rate(const Context<double>& c) const {
    return 6.0;
}

template<>
double reaction<pmb_synthesis>::active_rate(const Context<double>& c) const {
    return 6.0;
}

template<>
double reaction<pmb_degradation>::active_rate(const Context<double>& c) const {
    return 6.0;
}

/*
template<>
double reaction<mespb_dissociation>::active_rate(const Context<double>& c) const {
    return 6.0;
}

template<>
double reaction<mespb_dissociation2>::active_rate(const Context<double>& c) const {
    return 6.0;
}

template<>
double reaction<mespb_dissociation3>::active_rate(const Context<double>& c) const {
    return 6.0;
}

template<>
double reaction<mespb_association>::active_rate(const Context<double>& c) const {
    return 6.0;
}

template<>
double reaction<mespb_association2>::active_rate(const Context<double>& c) const {
    return 6.0;
}

template<>
double reaction<mespb_association3>::active_rate(const Context<double>& c) const {
    return 6.0;
}
*/
template<>
double reaction<ph11_degradation>::active_rate(const Context<double>& c) const {
    return 6.0;
}

template<>
double reaction<pmaa_degradation>::active_rate(const Context<double>& c) const {
    return 6.0;
}

template<>
double reaction<pmab_degradation>::active_rate(const Context<double>& c) const {
    return 6.0;
}

template<>
double reaction<pmbb_degradation>::active_rate(const Context<double>& c) const {
    return 6.0;
}

template<>
double reaction<mh1_synthesis>::active_rate(const Context<double>& c) const {
    return 6.0;
}

template<>
double reaction<mh1_degradation>::active_rate(const Context<double>& c) const {
    return 6.0;
}

template<>
double reaction<md_synthesis>::active_rate(const Context<double>& c) const {
    return 6.0;
}

template<>
double reaction<md_degradation>::active_rate(const Context<double>& c) const {
    return 6.0;
}

template<>
double reaction<mma_synthesis>::active_rate(const Context<double>& c) const {
    return 6.0;
}

template<>
double reaction<mma_degradation>::active_rate(const Context<double>& c) const {
    return 6.0;
}

template<>
double reaction<mmb_synthesis>::active_rate(const Context<double>& c) const {
    return 6.0;
}

template<>
double reaction<mmb_degradation>::active_rate(const Context<double>& c) const {
    return 6.0;
}

#endif // MODEL_IMPL_H
