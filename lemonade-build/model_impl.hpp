#ifndef MODEL_IMPL_H
#define MODEL_IMPL_H
#include "core/reaction.hpp"
#include "core/specie.hpp"
#include "core/model.hpp"
#include "sim/base.hpp"
#include <cstddef>

template<>
template<class Ctxt>
RATETYPE reaction<make_lemonade>::active_rate(const Ctxt& c) {
    return c.getRate(make_lemonade) * c.getCon(lemon)*c.getCon(sugar);
}

template<>
template<class Ctxt>
RATETYPE reaction<make_lemon>::active_rate(const Ctxt& c) {
    return (c.getRate(make_lemon) * (c.getCon(lemon) - (c.getCon(lemon) *c.getCritVal(rcrit_lemon))));
}

template<>
template<class Ctxt>
RATETYPE reaction<make_sugar>::active_rate(const Ctxt& c) {
    return (c.getRate(make_sugar) * (c.getCon(sugar) - c.getCritVal(rcrit_sugar)));
}

template<>
template<class Ctxt>
RATETYPE reaction<customer_purchase>::active_rate(const Ctxt& c) {
    return c.getRate(customer_purchase);
}

template<>
template<class Ctxt>
RATETYPE reaction<buy_lemons>::active_rate(const Ctxt& c) {
    return c.getRate(buy_lemons);
}

template<>
template<class Ctxt>
Real reaction<buy_sugar>::active_rate(const Ctxt& c) {
    return c.getRate(buy_sugar);
}

template<>
template<class Ctxt>
Real reaction<make_quality>::active_rate(const Ctxt& c) {
    return c.getRate(make_quality)*c.getCon(lemon)*c.getCon(sugar);
}

template<>
template<class Ctxt>
Real reaction<quality_to_satisfaction>::active_rate(const Ctxt& c) {
    return c.getRate(quality_to_satisfaction);
}

template<>
template<class Ctxt>
Real reaction<satisfaction_to_tips>::active_rate(const Ctxt& c) {
    return c.getRate(satisfaction_to_tips);
}
#endif // MODEL_IMPL_H
