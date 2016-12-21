#include "simulation.hpp"
#include <iostream>

int main() {
  model m;
  param_set ps;
  simulation s(m, ps);
  s.test_sim();
}
