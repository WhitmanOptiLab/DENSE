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
    return c.getRate(ph1_synthesis) * c.getCon(mh1);
}

template<>
RATETYPE reaction<ph1_degradation>::active_rate(const Context& c) const {
    return c.getRate(ph1_degradation) * c.getCon(ph1);
}

template<>
RATETYPE reaction<ph11_dissociation>::active_rate(const Context& c) const {
    return c.getRate(ph11_dissociation) * c.getCon(ph11);
}

template<>
RATETYPE reaction<ph11_association>::active_rate(const Context& c) const {
    return c.getRate(ph11_association) * c.getCon(ph1) * c.getCon(ph1);
}

template<>
RATETYPE reaction<pd_synthesis>::active_rate(const Context& c) const {
    return c.getRate(pd_synthesis) * c.getCon(md);
}

template<>
RATETYPE reaction<pd_degradation>::active_rate(const Context& c) const {
    return c.getRate(pd_degradation) * c.getCon(pd);
}

template<>
RATETYPE reaction<pm1_synthesis>::active_rate(const Context& c) const {
    return c.getRate(pm1_synthesis) * c.getCon(mm1);
}

template<>
RATETYPE reaction<pm1_degradation>::active_rate(const Context& c) const {
    return c.getRate(pm1_degradation) * c.getCon(pm1);
}

template<>
RATETYPE reaction<pm2_synthesis>::active_rate(const Context& c) const {
    return c.getRate(pm2_synthesis) * c.getCon(mm2);
}

template<>
RATETYPE reaction<pm2_degradation>::active_rate(const Context& c) const {
    return c.getRate(pm2_degradation) * c.getCon(pm2);
}

template<>
RATETYPE reaction<pm22_association>::active_rate(const Context& c) const {
    return c.getRate(pm22_association) * c.getCon(pm2) * c.getCon(pm2);
}

template<>
RATETYPE reaction<pm11_association>::active_rate(const Context& c) const {
    return c.getRate(pm11_association) * c.getCon(pm1) * c.getCon(pm1);
}

template<>
RATETYPE reaction<pm12_association>::active_rate(const Context& c) const {
    return c.getRate(pm12_association) * c.getCon(pm1) * c.getCon(pm2);
}

template<>
RATETYPE reaction<pm11_dissociation>::active_rate(const Context& c) const {
    return c.getRate(pm11_dissociation) * c.getCon(pm11);
}

template<>
RATETYPE reaction<pm12_dissociation>::active_rate(const Context& c) const {
    return c.getRate(pm12_dissociation) * c.getCon(pm12);
}

template<>
RATETYPE reaction<pm22_dissociation>::active_rate(const Context& c) const {
    return c.getRate(pm22_dissociation) * c.getCon(pm22);
}

template<>
RATETYPE reaction<ph11_degradation>::active_rate(const Context& c) const {
    return c.getRate(ph11_degradation) * c.getCon(ph11);
}

template<>
RATETYPE reaction<pm11_degradation>::active_rate(const Context& c) const {
    return c.getRate(pm11_degradation) * c.getCon(pm11);
}

template<>
RATETYPE reaction<pm12_degradation>::active_rate(const Context& c) const {
    return c.getRate(pm12_degradation) * c.getCon(pm12);
}

template<>
RATETYPE reaction<pm22_degradation>::active_rate(const Context& c) const {
    return c.getRate(pm22_degradation) * c.getCon(pm22);
}

template<>
RATETYPE reaction<mh1_synthesis>::active_rate(const Context& c) const {
    RATETYPE tdelta = c.calculateNeighborAvg(pd)/c.getCritVal(rcrit_pd);
    RATETYPE th1h1 = c.getCon(ph11)/c.getCritVal(rcrit_ph11);
    return c.getRate(mh1_synthesis) * (tdelta/(RATETYPE(1.0)+tdelta+(th1h1*th1h1)));
}

template<>
RATETYPE reaction<mh1_degradation>::active_rate(const Context& c) const {
    return c.getRate(mh1_degradation) * c.getCon(mh1);
}

template<>
RATETYPE reaction<md_synthesis>::active_rate(const Context& c) const {
    RATETYPE th1h1 = c.getCon(ph11)/c.getCritVal(rcrit_ph11);
    return c.getRate(md_synthesis) * (RATETYPE(1.0)/(RATETYPE(1.0)+(th1h1*th1h1)));
}

template<>
RATETYPE reaction<md_degradation>::active_rate(const Context& c) const {
    return c.getRate(md_degradation) * c.getCon(md);
}

template<>
RATETYPE reaction<mm1_synthesis>::active_rate(const Context& c) const {
    RATETYPE tdelta = c.calculateNeighborAvg(pd)/c.getCritVal(rcrit_pd);
    RATETYPE th1h1 = c.getCon(ph11)/c.getCritVal(rcrit_ph11);
    return c.getRate(mm1_synthesis) * ((RATETYPE(1.0) + tdelta)/(RATETYPE(1.0)+tdelta+(th1h1*th1h1)));
}

template<>
RATETYPE reaction<mm1_degradation>::active_rate(const Context& c) const {
    return c.getRate(mm1_degradation) * c.getCon(mm1);
}

template<>
RATETYPE reaction<mm2_synthesis>::active_rate(const Context& c) const {
    RATETYPE tdelta = c.calculateNeighborAvg(pd)/c.getCritVal(rcrit_pd);
    RATETYPE tm1m1 = c.getCon(pm11)/c.getCritVal(rcrit_pm11);
    RATETYPE tm1m2 = c.getCon(pm12)/c.getCritVal(rcrit_pm12);
    RATETYPE tm2m2 = c.getCon(pm22)/c.getCritVal(rcrit_pm22);
    RATETYPE denominator = RATETYPE(1.0) + tdelta + tm1m1*tm1m1 + tm1m2*tm1m2 + tm2m2*tm2m2;
    return c.getRate(mm2_synthesis) * (RATETYPE(1.0)+tdelta)/denominator;
}

template<>
RATETYPE reaction<mm2_degradation>::active_rate(const Context& c) const {
    return c.getRate(mm2_degradation) * c.getCon(mm2);
}

#endif // MODEL_IMPL_H
