// In this header file, define your model!
// This includes functions to describe each reaction.
// Make sure that you've first completed reaction_list.h and specie_list.h
#ifndef MODEL_IMPL_H
#define MODEL_IMPL_H
#include "reaction.hpp"
#include "specie.hpp"
#include "model.hpp"
#include "context.hpp"
#include <cstddef>

// Next, define all of your reaction rate functions
// For example, if you enumerated a reaction R_ONE, you should declare a 
//   function like this:
//
// RATETYPE reaction<R_ONE>::active_rate(const Context c) const { return 6.0; }
// 
// Or, for a more interesting reaction rate, you might do something like
// 
// RATETYPE reaction<R_TWO>::active_rate(const Context c) const {
//   return c.rate[R_TWO] * c.concentration[SPECIE_ONE] * 
//                                   c.neighbors.concentration[SPECIE_TWO];
// }
/*
template<>
RATETYPE reaction<one>::active_rate(const Context& c) const {
  return 6.0;
}

template<>
RATETYPE reaction<two>::active_rate(const Context& c) const {
  return 3.0;
}

template<>
RATETYPE reaction<three>::active_rate(const Context& c) const {
  return 8.0;
}
*/

template<>
RATETYPE reaction<ph1_synthesis>::active_rate(const Context& c) const {
    return 6.0;
}

template<>
RATETYPE reaction<ph1_degradation>::active_rate(const Context& c) const {
    return 6.0;
}

template<>
RATETYPE reaction<ph11_dissociation>::active_rate(const Context& c) const {
    return 6.0;
}

template<>
RATETYPE reaction<ph11_association>::active_rate(const Context& c) const {
    return 6.0;
}

template<>
RATETYPE reaction<pd_synthesis>::active_rate(const Context& c) const {
    return 6.0;
}

template<>
RATETYPE reaction<pd_degradation>::active_rate(const Context& c) const {
    return 6.0;
}

template<>
RATETYPE reaction<pm1_synthesis>::active_rate(const Context& c) const {
    return 6.0;
}

template<>
RATETYPE reaction<pm1_degradation>::active_rate(const Context& c) const {
    return 6.0;
}

template<>
RATETYPE reaction<pm22_association>::active_rate(const Context& c) const {
    return 6.0;
}

template<>
RATETYPE reaction<pm11_association>::active_rate(const Context& c) const {
    return 6.0;
}

template<>
RATETYPE reaction<pm12_association>::active_rate(const Context& c) const {
    return 6.0;
}

template<>
RATETYPE reaction<pm11_dissociation>::active_rate(const Context& c) const {
    return 6.0;
}

template<>
RATETYPE reaction<pm12_dissociation>::active_rate(const Context& c) const {
    return 6.0;
}

template<>
RATETYPE reaction<pm22_dissociation>::active_rate(const Context& c) const {
    return 6.0;
}

template<>
RATETYPE reaction<pm2_synthesis>::active_rate(const Context& c) const {
    return 6.0;
}

template<>
RATETYPE reaction<pm2_degradation>::active_rate(const Context& c) const {
    return 6.0;
}

template<>
RATETYPE reaction<ph11_degradation>::active_rate(const Context& c) const {
    return 6.0;
}

template<>
RATETYPE reaction<pm11_degradation>::active_rate(const Context& c) const {
    return 6.0;
}

template<>
RATETYPE reaction<pm12_degradation>::active_rate(const Context& c) const {
    return 6.0;
}

template<>
RATETYPE reaction<pm22_degradation>::active_rate(const Context& c) const {
    return 6.0;
}

template<>
RATETYPE reaction<mh1_synthesis>::active_rate(const Context& c) const {
    return 6.0;
}

template<>
RATETYPE reaction<mh1_degradation>::active_rate(const Context& c) const {
    return 6.0;
}

template<>
RATETYPE reaction<md_synthesis>::active_rate(const Context& c) const {
    return 6.0;
}

template<>
RATETYPE reaction<md_degradation>::active_rate(const Context& c) const {
    return 6.0;
}

template<>
RATETYPE reaction<mm1_synthesis>::active_rate(const Context& c) const {
    return 6.0;
}

template<>
RATETYPE reaction<mm1_degradation>::active_rate(const Context& c) const {
    return 6.0;
}

template<>
RATETYPE reaction<mm2_synthesis>::active_rate(const Context& c) const {
    return 6.0;
}

template<>
RATETYPE reaction<mm2_degradation>::active_rate(const Context& c) const {
    return 6.0;
}

#endif // MODEL_IMPL_H
