#ifndef CORE_MODEL_HPP
#define CORE_MODEL_HPP

#include <stdexcept>

#include "reaction.hpp"
#include "specie.hpp"


class model{
public:
    model() {}

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

    reaction_base const& getReaction(reaction_id rid) const {
        switch (rid) {
            #define REACTION(name) \
            case name: return reaction_##name;
            #include "reactions_list.hpp"
            #undef REACTION
            default: throw std::out_of_range("Invalid reaction ID: " + std::to_string(rid));
        }
    }

    #define REACTION(name) reaction<name> reaction_##name;
    #include "reactions_list.hpp"
    #undef REACTION
};

#endif
