// In this header file, define your model!
// This includes functions to describe each reaction.
// Make sure that you've first completed reaction_list.h and specie_list.h







// First, define all of your reaction rate functions
// For example, if you enumerated a reaction R_ONE, you should declare a 
//   function like this:
//
// double reaction<R_ONE>::active_rate(const Context c) { return 6.0; }
// 
// Or, for a more interesting reaction rate, you might do something like
// 
// double reaction<R_TWO>::active_rate(const Context c) {
//   return c.rate[R_TWO] * c.concentration[SPECIE_ONE] * 
//                                   c.neighbors.concentration[SPECIE_TWO];
// }
#ifndef REACTION_RATES_H
#define REACTION_RATES_H
#include "reaction.hpp"
#include "model.hpp"

template<>
double reaction<one>::active_rate(const Context& c) {
  return 6.0;
}

template<>
double reaction<two>::active_rate(const Context& c) {
  return 3.0;
}

template<>
double reaction<three>::active_rate(const Context& c) {
  return 8.0;
}

#endif // REACTION_RATES_H
