#ifndef MODEL_IMPL_H
#define MODEL_IMPL_H
#include "core/reaction.hpp"
#include "core/specie.hpp"
#include "core/model.hpp"
#include "sim/base.hpp"
#include <cstddef>


template<>
template<class Ctxt>
RATETYPE reaction<system_1>::active_rate(const Ctxt& c) {
	return c.getCon(x_1) - c.getCon(x_2);
}

template<>
template<class Ctxt>
RATETYPE reaction<system_2>::active_rate(const Ctxt& c) {
	return -1*c.getCon(x_1) - 0.5*c.getCon(x_2);
}
#endif // MODEL_IMPL_H
