
#include "simulation.hpp"

#include "model_impl.hpp"

#include <iostream>

class Context {
 //Nothing to report
};

//A quick test case to make sure all reaction rates are defined by link time
void simulation::test_sim() {
  Context c;

  double sum_rates = 0.0;
#define REACTION(name) sum_rates += _model.name.active_rate(c);
  std::cout << "If you're seeing this, simulation.cpp compiles correctly:" 
            << sum_rates << std::endl;
}
