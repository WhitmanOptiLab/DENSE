#ifndef CORE_MODEL_HPP
#define CORE_MODEL_HPP

#include <stdexcept>

#include "reaction.hpp"
#include "specie.hpp"

namespace dense {

class model{
public:
    model() = delete;

    #define REACTION(name) static CUDA_MANAGED reaction<name> reaction_##name;
    #include "reactions_list.hpp"
    #undef REACTION

    static delay_reaction_id getDelayReactionId(reaction_id rid) {
        switch (rid) {
            #define REACTION(name)
            #define DELAY_REACTION(name) \
            case name : return dreact_##name;
            #include "reactions_list.hpp"
            #undef REACTION
            #undef DELAY_REACTION
            default: return NUM_DELAY_REACTIONS;
        }
    }

    static reaction_base const& getReaction(reaction_id rid) {
        switch (rid) {
            #define REACTION(name) \
            case name: return reaction_##name;
            #include "reactions_list.hpp"
            #undef REACTION
            default: throw std::out_of_range("Invalid reaction ID: " + std::to_string(static_cast<unsigned>(rid)));
        }
    }

    template <typename T>
    static Real active_rate(reaction_id id, T context) {
      switch (id) {
        #define REACTION(name)\
          case name: return reaction_##name.active_rate(context);
        #include "reactions_list.hpp"
        #undef REACTION
        default: throw std::out_of_range("Invalid reaction ID: " + std::to_string(static_cast<unsigned>(id)));
      }
    }

};

}

#endif
