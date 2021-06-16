#include "random_selector.hpp"

#include <iostream>
#include <random>

using namespace dense::stochastic;

int main() {
  std::default_random_engine generator;
  std::vector<float> weights = {4, 3, 2, 1};

  nonuniform_int_distribution<int> selector(weights);
  std::cout << "Constructed" << std::endl;
  std::vector<unsigned int> counts(weights.size());

  for (int i = 0; i < 100000; i++) {
    int index = selector(generator);
    if (index < 0 || index >= weights.size()) {
      std::cout << "Error, index out of range" << std::endl;
    } else {
      counts[index]++;
    }
  }
  for (auto c : counts) {
    std::cout << c << ',';
  }
  std::cout << std::endl;
}
