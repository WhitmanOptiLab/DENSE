#include "simulation.hpp"
#include <iostream>

int main() {
  model m(false, false);
  param_set ps;
  simulation s(m, ps, 1, 1, 1.0);
  s.test_sim();
}
