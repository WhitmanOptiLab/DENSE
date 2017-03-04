#include "simulation.hpp"
#include <iostream>

int main() {
  model m;
  param_set ps;
  simulation s(m, ps, 1, 1);
  s.test_sim();
}
