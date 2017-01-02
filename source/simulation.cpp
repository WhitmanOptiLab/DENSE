
#include "simulation.hpp"

#include "model_impl.hpp"

#include <iostream>

//declare reaction inits here
#define REACTION(name) \
  template<> \
  reaction< name >::reaction() : \
    num_inputs(num_inputs_##name), num_outputs(num_outputs_##name), \
    in_counts(in_counts_##name), inputs(inputs_##name), \
    out_counts(out_counts_##name), outputs(outputs_##name) {}
LIST_OF_REACTIONS
#undef REACTION

//A quick test case to make sure all reaction rates are defined by link time
void simulation::test_sim() {
  Context<double> c;

  double sum_rates = 0.0;
#define REACTION(name) sum_rates += _model.reaction_##name.active_rate(c);
  LIST_OF_REACTIONS
#undef REACTION
  std::cout << "If you're seeing this, simulation.cpp compiles correctly:" 
            << sum_rates << std::endl;
}

/*
void simulation::model(){
    Context<double> contexts[]= {};
    int j;
    int baby_j; 
    bool past_induction[6];
    bool past_recovery[6];
    Context<double> c;
    
    double sum_rates = 0.0;
    
    
#define REACTION(name) sum_rates += _model.reaction_##name.active_rate(c);
    LIST_OF_REACTIONS
#undef REACTION

}*/
