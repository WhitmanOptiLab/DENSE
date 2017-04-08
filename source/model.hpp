#ifndef MODEL_HPP
#define MODEL_HPP
#include <cstddef>

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
    model(bool using_gradients, bool using_perturb) :
      _using_perturb(using_perturb),
      _using_gradients(using_gradients) {
      for (int i = 0; i < NUM_REACTIONS; i++) {
        factors_perturb[i] = 0.0;
        _has_gradient[i] = false;
        factors_gradient[i] = NULL;
      }
    }
    
#define REACTION(name) reaction<name> reaction_##name;
    #include "reactions_list.hpp"
#undef REACTION
    
    
};

#endif

