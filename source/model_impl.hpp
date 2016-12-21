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
double reaction<mespa_synthesis>::active_rate(const Context<double>& c) const {
    return 6.0;
}

template<>
double reaction<mepsa_degradation>::active_rate(const Context<double>& c) const {
    return 6.0;
}

template<>
double reaction<mespa_dissociation>::active_rate(const Context<double>& c) const {
    return 6.0;
}

template<>
double reaction<mespa_dissociation2>::active_rate(const Context<double>& c) const {
    return 6.0;
}

template<>
double reaction<mespa_dissociation3>::active_rate(const Context<double>& c) const {
    return 6.0;
}

template<>
double reaction<mespa_association>::active_rate(const Context<double>& c) const {
    return 6.0;
}

template<>
double reaction<mespa_association2>::active_rate(const Context<double>& c) const {
    return 6.0;
}

template<>
double reaction<mespa_association3>::active_rate(const Context<double>& c) const {
    return 6.0;
}

template<>
double reaction<mespb_synthesis>::active_rate(const Context<double>& c) const {
    return 6.0;
}

template<>
double reaction<mepsb_degradation>::active_rate(const Context<double>& c) const {
    return 6.0;
}

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

template<>
double reaction<ph11_degradation>::active_rate(const Context<double>& c) const {
    return 6.0;
}

template<>
double reaction<pm11_degradation>::active_rate(const Context<double>& c) const {
    return 6.0;
}

template<>
double reaction<pm12_degradation>::active_rate(const Context<double>& c) const {
    return 6.0;
}

template<>
double reaction<pm22_degradation>::active_rate(const Context<double>& c) const {
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
double reaction<mm1_synthesis>::active_rate(const Context<double>& c) const {
    return 6.0;
}

template<>
double reaction<mm1_degradation>::active_rate(const Context<double>& c) const {
    return 6.0;
}

template<>
double reaction<mm2_synthesis>::active_rate(const Context<double>& c) const {
    return 6.0;
}

template<>
double reaction<mm2_degradation>::active_rate(const Context<double>& c) const {
    return 6.0;
}

#endif // MODEL_IMPL_H
