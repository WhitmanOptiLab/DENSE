#ifndef CORE_MODEL_HPP
#define CORE_MODEL_HPP
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <string>
#include "reaction.hpp"
#include "specie.hpp"

using namespace std;

class model{
  private:

    /* rates contains the rates specified by the current parameter set as well as perturbation and gradient data
     notes:
     There should be only one instance of rates at any time.
     rates_active is the final, active rates that should be used in the simulation.
     todo:
     */
       
  public:
    
    //currently in use
    bool _using_perturb;
    RATETYPE factors_perturb[NUM_REACTIONS]; // Perturbations (as percentages with 1=100%) taken from the perturbations input file
    bool _using_gradients; // Whether or not any rates have specified perturbations
    RATETYPE* factors_gradient[NUM_REACTIONS];
    bool _has_gradient[NUM_REACTIONS]; // Whether each rate has a specified gradient

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

    const reaction_base& getReaction(reaction_id rid) const {
        switch (rid) {
            #define REACTION(name) \
            case name: return reaction_##name;
            #include "reactions_list.hpp"
            #undef REACTION
            default: cout<<"exiting"<<endl; exit (-1);
        }
    }


    
    #define REACTION(name) reaction<name> reaction_##name;
    #include "reactions_list.hpp"
    #undef REACTION

    model(const string& pcfGradientFile, const string& pcfPerturbFile,
            const int& pcfTotalWidth);
};

#endif

