#include "simulation.hpp"
#include <iostream>

int main() {
  model m;
  parameter_set ps;
  simulation s(m, ps);
  s.test_sim();
}
