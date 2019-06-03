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
#include <cmath>

/*

Define all of your reaction rate functions in `model_impl.hpp`.
For example, if you enumerated a reaction `R_ONE`, you should declare a
   function like this:

 RATETYPE reaction<R_ONE>::active_rate(const Ctxt& c) { return 6.0; }


Or, for a more interesting reaction rate, you might do something like:


 RATETYPE reaction<R_TWO>::active_rate(const Ctxt& c) {
   return c.getRate(R_TWO) * c.getCon(SPECIE_ONE) * c.neighbors.calculateNeighborAvg(SPECIE_TWO);
 }

Refer to the Context API (Section ) for instructions on how to get delays
   and critical values for more complex reaction rate functions.

*/
template<>
template<class Ctxt>
RATETYPE reaction<activator_diffusion_rate>::active_rate(const Ctxt& c) {
    if(c.calculateNeighborAvg(activator,1)>c.getCon(activator)){
        return c.getRate(activator_diffusion_rate)*c.getCon(activator);
    }else{
        return c.getRate(activator_diffusion_rate)*c.getCon(activator)-(c.getCon(activator_dt));
    }
}

template<>
template<class Ctxt>
RATETYPE reaction<inhibitor_diffusion_rate>::active_rate(const Ctxt& c) {
    if(c.calculateNeighborAvg(inhibitor,1)>c.getCon(inhibitor)){
        return c.getRate(inhibitor_diffusion_rate)*c.getCon(inhibitor);
    }else{
        return c.getRate(inhibitor_diffusion_rate)*c.getCon(inhibitor)-(c.getCon(inhibitor_dt));
    }
}

template<>
template<class Ctxt>
RATETYPE reaction<activator_synthesis>::active_rate(const Ctxt& c) {
    return c.getRate(activator_synthesis)+c.getCon(activator_dt);
}

template<>
template<class Ctxt>
RATETYPE reaction<inhibitor_synthesis>::active_rate(const Ctxt& c) {
    return c.getRate(inhibitor_synthesis)+c.getCon(inhibitor_dt);
}

template<>
template<class Ctxt>
RATETYPE reaction<activator_dCon>::active_rate(const Ctxt& c) {
    return c.getRate(activator_dCon)+c.getCon(activator_dt);
}

template<>
template<class Ctxt>
RATETYPE reaction<inhibitor_dCon>::active_rate(const Ctxt& c) {
    return c.getRate(inhibitor_dCon)+c.getCon(inhibitor_dt);
}

#endif // MODEL_IMPL_H
