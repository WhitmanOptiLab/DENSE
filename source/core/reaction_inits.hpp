#ifndef REACTION_INITS_HPP
#define REACTION_INITS_HPP
#include "core/model.hpp"

#define REACTION(name) \
  template<> \
  reaction< name >::reaction() : \
    reaction_base( num_deltas_##name, \
    deltas_##name, delta_ids_##name){}
#include "reactions_list.hpp"
#undef REACTION

#define REACTION(name) \
  reaction<name> dense::model::reaction_##name{};
#include "reactions_list.hpp"
#undef REACTION

#endif // REACTION_INITS_HPP
