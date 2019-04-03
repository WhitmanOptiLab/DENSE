// In this header file, define your model!
// This includes functions to describe each reaction.
// Make sure that you've first completed reaction_list.h and specie_list.h
#ifndef MODEL_IMPL_H
#define MODEL_IMPL_H
#include "core/reaction.hpp"
#include "core/specie.hpp"
#include "core/model.hpp"
#include "sim/base.hpp"
//#include "context.hpp"
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
Real reaction<infection>::active_rate(Context const& c) {
    return c.getRate(infection) * c.getCon(vulnerable) * c.getCon(infected);
}

template<>
template<typename Context>
Real reaction<immunization>::active_rate(Context const& c) {
    return c.getRate(immunization) * c.getCon(vulnerable);
}

template<>
template<typename Context>
Real reaction<influx>::active_rate(Context const& c) {
    return c.getRate(influx) * (c.getCon(vulnerable) + c.getCon(immune));
}

template<>
template<class Context>
Real reaction<recovery>::active_rate(Context const& c) {
    return c.getRate(recovery) * c.getCon(infected);
}

template<>
template<class Context>
Real reaction<treatment>::active_rate(Context const& c) {
    return c.getRate(treatment) * c.getCon(infected);
}

template<>
template<class Context>
Real reaction<death>::active_rate(Context const& c) {
    return c.getRate(death) * c.getCon(infected);
}

#endif // MODEL_IMPL_H
