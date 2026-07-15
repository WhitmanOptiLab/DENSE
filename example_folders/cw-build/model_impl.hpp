#ifndef MODEL_IMPL_H
#define MODEL_IMPL_H
#include "core/reaction.hpp"
#include "core/specie.hpp"
#include "core/model.hpp"
#include "sim/base.hpp"
#include <cstddef>

// magnitude of speed*a*p  (a is negative, so -a is positive; delta=-1 applies the sign)
template<>
template<class Ctxt>
RATETYPE reaction<demand_slope>::active_rate(const Ctxt& c) {
    return -c.getCritVal(rcrit_speed) * c.getCritVal(rcrit_a) * c.getCon(P);
}

// speed*b
template<>
template<class Ctxt>
RATETYPE reaction<demand_base>::active_rate(const Ctxt& c) {
    return c.getCritVal(rcrit_speed) * c.getCritVal(rcrit_b);
}

// magnitude of speed*c*p_delayed; delta=-1 applies the sign
template<>
template<class Ctxt>
RATETYPE reaction<supply_slope>::active_rate(const Ctxt& c) {
    RATETYPE p_delayed = c.getCon(P, c.getDelay(dreact_supply_slope));
    return c.getCritVal(rcrit_speed) * c.getCritVal(rcrit_c) * p_delayed;
}

// magnitude of speed*d; delta=-1 applies the sign
template<>
template<class Ctxt>
RATETYPE reaction<supply_base>::active_rate(const Ctxt& c) {
    return c.getCritVal(rcrit_speed) * c.getCritVal(rcrit_d);
}

#endif // MODEL_IMPL_H