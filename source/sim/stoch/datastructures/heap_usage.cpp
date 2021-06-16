#include "heap_tree.hpp"
#include <iostream>
#include <algorithm>


using namespace dense::stochastic;


int main() {
  std::vector<float> items = {10, 4, 3, 42, 15, 82, 95, 49, 2};
  heap_tree<float, std::greater<float> > ht(items);

  std::sort(items.rbegin(), items.rend());

  for (auto i : items) {
    std::cout << i << ", " << ht.top() << std::endl;
    ht.pop();
  }
  return 0;
}
