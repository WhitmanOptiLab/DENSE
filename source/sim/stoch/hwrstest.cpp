#include "heap_random_selector.hpp"

#include <iostream>
#include <random>

using namespace dense::stochastic;

int main() {
  std::default_random_engine generator;
  std::vector<float> weights = {1, 2, 3, 4};

  heap_random_selector<int> selector(weights);
  std::cout << "Constructed" << std::endl;
  std::vector<unsigned int> counts(weights.size());

  while (!selector.empty()) {
    counts = std::vector<unsigned int>(weights.size());
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
    std::cout << std::endl;

    auto top = selector.top();
    
    std::cout << std::get<0>(top) << ", "
              << std::get<1>(top) << ", "
              << std::get<2>(top) << std::endl;

    selector.pop();
  }
}
