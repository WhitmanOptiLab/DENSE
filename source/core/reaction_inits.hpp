#ifndef REACTION_INITS_HPP
#define REACTION_INITS_HPP

#include "util/common_utils.hpp"
#include "core/model.hpp"

#define REACTION(name) \
  template<> \
  reaction< name >::reaction() : \
    reaction_base( num_inputs_##name, num_outputs_##name, \
    num_factors_##name, in_counts_##name, out_counts_##name, \
    inputs_##name, outputs_##name, factors_##name){}
#include "reactions_list.hpp"
#undef REACTION

#endif // REACTION_INITS_HPP
