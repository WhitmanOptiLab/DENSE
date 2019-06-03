// In this header file, define your model!
// This includes functions to describe each reaction.
// Make sure that you've first completed reaction_list.h and specie_list.h
#ifndef MODEL_IMPL_H
#define MODEL_IMPL_H
#include "core/reaction.hpp"
#include "core/specie.hpp"
#include "core/model.hpp"
#include "sim/base.hpp"
#include <cstddef>

/*

Define all of your reaction rate functions in `model_impl.hpp`.
For example, if you enumerated a reaction `R_ONE`, you should declare a
   function like this:

 RATETYPE reaction<R_ONE>::active_rate(Ctxt const& c) const { return 6.0; }


Or, for a more interesting reaction rate, you might do something like:


 RATETYPE reaction<R_TWO>::active_rate(Ctxt const& c) const {
   return c.getRate(R_TWO) * c.getCon(SPECIE_ONE) * c.neighbors.calculateNeighborAvg(SPECIE_TWO);
 }

Refer to the Context API (Section ) for instructions on how to get delays
   and critical values for more complex reaction rate functions.

*/

template<>
template<typename Context>
Real reaction<red_warming>::active_rate(Context const& c) {
  return (c.getCritVal(rcrit_red_room_dT) + c.getCon(red_room_dT) - c.getCon(red_box_T));
}

template<>
template<typename Context>
Real reaction<green_warming>::active_rate(Context const& c) {
  return (c.getCritVal(rcrit_green_room_dT) + c.getCon(green_room_dT) - c.getCon(green_box_T));
}

template<>
template<typename Context>
Real reaction<blue_warming>::active_rate(Context const& c) {
  return (c.getCritVal(rcrit_blue_room_dT) + c.getCon(blue_room_dT) - c.getCon(blue_box_T));
}

template<>
template<typename Context>
Real reaction<red_green_diffusion>::active_rate(Context const& c) {
  return (c.getCritVal(rcrit_red_room_dT) + c.getCon(red_room_dT)) - (c.getCritVal(rcrit_green_room_dT) + c.getCon(green_room_dT));
}

template<>
template<typename Context>
Real reaction<green_blue_diffusion>::active_rate(Context const& c) {
  return (c.getCritVal(rcrit_green_room_dT) + c.getCon(green_room_dT)) - (c.getCritVal(rcrit_blue_room_dT) + c.getCon(blue_room_dT));
}

#endif // MODEL_IMPL_H
