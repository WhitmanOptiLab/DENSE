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
RATETYPE reaction<ph1_synthesis>::active_rate(const Ctxt& c) {
    return c.getRate(ph1_synthesis) * c.getCon(mh1,c.getDelay(dreact_ph1_synthesis));
}

template<>
template<class Ctxt>
RATETYPE reaction<ph1_degradation>::active_rate(const Ctxt& c) {
    return c.getRate(ph1_degradation) * c.getCon(ph1);
}

template<>
template<class Ctxt>
RATETYPE reaction<ph7_synthesis>::active_rate(const Ctxt& c) {
    return c.getRate(ph7_synthesis) * c.getCon(mh7,c.getDelay(dreact_ph7_synthesis));
}

template<>
template<class Ctxt>
Real reaction<ph7_degradation>::active_rate (const Ctxt& c) {
    return c.getRate(ph7_degradation) * c.getCon(ph7);
}

template<>
template<class Ctxt>
RATETYPE reaction<ph13_synthesis>::active_rate(const Ctxt& c) {
    return c.getRate(ph13_synthesis) * c.getCon(mh13,c.getDelay(dreact_ph13_synthesis));
}

template<>
template<class Ctxt>
RATETYPE reaction<ph13_degradation>::active_rate(const Ctxt& c) {
    return c.getRate(ph13_degradation) * c.getCon(ph13);
}

template<>
template<class Ctxt>
RATETYPE reaction<pd_synthesis>::active_rate(const Ctxt& c) {
    return c.getRate(pd_synthesis) * c.getCon(md,c.getDelay(dreact_pd_synthesis));
}

template<>
template<class Ctxt>
RATETYPE reaction<pd_degradation>::active_rate(const Ctxt& c) {
    return c.getRate(pd_degradation) * c.getCon(pd);
}

template<>
template<class Ctxt>
RATETYPE reaction<ph11_association>::active_rate(const Ctxt& c) {
    return c.getRate(ph11_association) * c.getCon(ph1) * c.getCon(ph1);
}

template<>
template<class Ctxt>
RATETYPE reaction<ph77_association>::active_rate(const Ctxt& c) {
    return c.getRate(ph77_association) * c.getCon(ph7) * c.getCon(ph7);
}

template<>
template<class Ctxt>
RATETYPE reaction<ph1313_association>::active_rate(const Ctxt& c) {
    return c.getRate(ph1313_association) * c.getCon(ph13) * c.getCon(ph13);
}

template<>
template<class Ctxt>
RATETYPE reaction<ph17_association>::active_rate(const Ctxt& c) {
    return c.getRate(ph17_association) * c.getCon(ph1) * c.getCon(ph7);
}

template<>
template<class Ctxt>
RATETYPE reaction<ph113_association>::active_rate(const Ctxt& c) {
    return c.getRate(ph113_association) * c.getCon(ph1) * c.getCon(ph13);
}

template<>
template<class Ctxt>
RATETYPE reaction<ph713_association>::active_rate(const Ctxt& c) {
    return c.getRate(ph713_association) * c.getCon(ph13) * c.getCon(ph7);
}

template<>
template<class Ctxt>
RATETYPE reaction<ph11_dissociation>::active_rate(const Ctxt& c) {
    return c.getRate(ph11_dissociation) * c.getCon(ph11);
}

template<>
template<class Ctxt>
RATETYPE reaction<ph77_dissociation>::active_rate(const Ctxt& c) {
    return c.getRate(ph77_dissociation) * c.getCon(ph77);
}

template<>
template<class Ctxt>
RATETYPE reaction<ph1313_dissociation>::active_rate(const Ctxt& c) {
    return c.getRate(ph1313_dissociation) * c.getCon(ph1313);
}

template<>
template<class Ctxt>
RATETYPE reaction<ph17_dissociation>::active_rate(const Ctxt& c) {
    return c.getRate(ph17_dissociation) * c.getCon(ph17);
}

template<>
template<class Ctxt>
RATETYPE reaction<ph113_dissociation>::active_rate(const Ctxt& c) {
    return c.getRate(ph113_dissociation) * c.getCon(ph113);
}

template<>
template<class Ctxt>
RATETYPE reaction<ph713_dissociation>::active_rate(const Ctxt& c) {
    return c.getRate(ph713_dissociation) * c.getCon(ph713);
}

template<>
template<class Ctxt>
RATETYPE reaction<ph11_degradation>::active_rate(const Ctxt& c) {
    return c.getRate(ph11_degradation) * c.getCon(ph11);
}

template<>
template<class Ctxt>
RATETYPE reaction<ph77_degradation>::active_rate(const Ctxt& c) {
    return c.getRate(ph77_degradation) * c.getCon(ph77);
}

template<>
template<class Ctxt>
RATETYPE reaction<ph1313_degradation>::active_rate(const Ctxt& c) {
    return c.getRate(ph1313_degradation) * c.getCon(ph1313);
}

template<>
template<class Ctxt>
RATETYPE reaction<ph17_degradation>::active_rate(const Ctxt& c) {
    return c.getRate(ph17_degradation) * c.getCon(ph17);
}

template<>
template<class Ctxt>
RATETYPE reaction<ph113_degradation>::active_rate(const Ctxt& c) {
    return c.getRate(ph113_degradation) * c.getCon(ph113);
}

template<>
template<class Ctxt>
RATETYPE reaction<ph713_degradation>::active_rate(const Ctxt& c) {
    return c.getRate(ph713_degradation) * c.getCon(ph713);
}

template<>
template<class Ctxt>
RATETYPE reaction<mh1_synthesis>::active_rate(const Ctxt& c) {
    RATETYPE tdelta = c.calculateNeighborAvg(pd,c.getDelay(dreact_mh1_synthesis))/c.getCritVal(rcrit_pd);
    RATETYPE th1h1 = c.getCon(ph11,c.getDelay(dreact_mh1_synthesis))/c.getCritVal(rcrit_ph11);
    RATETYPE th7h13 = c.getCon(ph713,c.getDelay(dreact_mh1_synthesis))/c.getCritVal(rcrit_ph713);
    return c.getRate(mh1_synthesis) * ((RATETYPE(1.0)+tdelta)/(RATETYPE(1.0)+tdelta+(th1h1*th1h1)+(th7h13*th7h13)));
}

template<>
template<class Ctxt>
RATETYPE reaction<mh1_degradation>::active_rate(const Ctxt& c) {
    return c.getRate(mh1_degradation) * c.getCon(mh1);
}

template<>
template<class Ctxt>
RATETYPE reaction<md_synthesis>::active_rate(const Ctxt& c) {
    RATETYPE th1h1 = c.getCon(ph11,c.getDelay(dreact_md_synthesis))/c.getCritVal(rcrit_ph11);
    RATETYPE th7h13 = c.getCon(ph713,c.getDelay(dreact_md_synthesis))/c.getCritVal(rcrit_ph713);
    return c.getRate(md_synthesis) * (RATETYPE(1.0)/(RATETYPE(1.0)+(th1h1*th1h1)+(th7h13*th7h13)));
}

template<>
template<class Ctxt>
RATETYPE reaction<md_degradation>::active_rate(const Ctxt& c) {
    return c.getRate(md_degradation) * c.getCon(md);
}

template<>
template<class Ctxt>
RATETYPE reaction<mh7_synthesis>::active_rate(const Ctxt& c) {
    RATETYPE tdelta = c.calculateNeighborAvg(pd,c.getDelay(dreact_mh7_synthesis))/c.getCritVal(rcrit_pd);
    RATETYPE th1h1 = c.getCon(ph11,c.getDelay(dreact_mh7_synthesis))/c.getCritVal(rcrit_ph11);
    RATETYPE th7h13 = c.getCon(ph713,c.getDelay(dreact_mh7_synthesis))/c.getCritVal(rcrit_ph713);
    return c.getRate(mh7_synthesis) * ((RATETYPE(1.0)+tdelta)/(RATETYPE(1.0)+tdelta+(th1h1*th1h1)+(th7h13*th7h13)));}

template<>
template<class Ctxt>
RATETYPE reaction<mh7_degradation>::active_rate(const Ctxt& c) {
    return c.getRate(mh7_degradation) * c.getCon(mh7);
}

template<>
template<class Ctxt>
RATETYPE reaction<mh13_synthesis>::active_rate(const Ctxt& c) {
    //RATETYPE tdelta = c.calculateNeighborAvg(pd)/c.getCritVal(rcrit_pd);
    //RATETYPE th1h1 = c.getCon(ph11)/c.getCritVal(rcrit_ph11);
    //RATETYPE th7h13 = c.getCon(ph713)/c.getCritVal(rcrit_ph713);
    //return c.getRate(mh13_synthesis) * ((RATETYPE(1.0)+tdelta)/(RATETYPE(1.0)+tdelta+(th1h1*th1h1)+(th7h13*th7h13)));
    return c.getRate(mh13_synthesis);
}

template<>
template<class Ctxt>
RATETYPE reaction<mh13_degradation>::active_rate(const Ctxt& c) {
    return c.getRate(mh13_degradation) * c.getCon(mh13);
}

#endif // MODEL_IMPL_H
