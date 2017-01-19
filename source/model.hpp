#ifndef MODEL_HPP
#define MODEL_HPP

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
    double factors_perturb[NUM_REACTIONS]; // Perturbations (as percentages with 1=100%) taken from the perturbations input file
    bool using_gradients; // Whether or not any rates have specified perturbations
    int width; // The total width of the simulation
    bool has_gradient[NUM_REACTIONS]; // Whether each rate has a specified gradient
    int cells; // The total number of cells in the simulation
    
  public:
    //array2D<double>  factors_gradient;
#define REACTION(name) reaction<name> reaction_##name;
    #include "reaction_list.hpp"
#undef REACTION
    
    
};

#endif

