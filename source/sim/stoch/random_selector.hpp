#ifndef RANDOM_SELECTOR
#define RANDOM_SELECTOR
#include "completetree.hpp"
#include "utility/numerics.hpp"
#include <limits>
#include <utility>
#include <random>

namespace dense {
namespace stochastic {


//Class to randomly select an index where each index's probability of being 
//  selected is weighted by a given vector.  
template <class IntType = int, size_t precision = std::numeric_limits<Real>::digits>
class nonuniform_int_distribution {
 public:
  //Weights can be of any type, but most be convertable to Real values
  nonuniform_int_distribution() = delete;
  nonuniform_int_distribution(IntType max_size) : total_weight{0.0}, _tree(max_size)  {};
  template<typename T>
  nonuniform_int_distribution(const std::vector<T>& weights) :
    total_weight{0.0},
    _tree{IntType(weights.size())}
  {
    for (auto w : weights) {
      _tree.emplace(Real(w), 0.0);
    }
    total_weight = sum_weights(_tree.root());
  }

  template<class URNG>
  IntType operator()(URNG& g) {

    Real target = std::generate_canonical<Real, precision, URNG>(g)*total_weight;
    std::cout << target << ' ' << total_weight << '\n';
    IntType node = _tree.root();
    while(_tree[_tree.left_of(node)].second > target ||
             _tree[node].first + _tree[_tree.left_of(node)].second < target) {
      if (_tree[_tree.left_of(node)].second > target) {
        node = _tree.left_of(node);
      } else {
        target -= _tree[node].first + _tree[_tree.left_of(node)].second;
        node = _tree.right_of(node);
      }
    }
    return node;
  }

 private:
  Real sum_weights(IntType i) {
    if (i > _tree.last()) return 0.0f;
    _tree[i].second =
      sum_weights(_tree.left_of(i)) + sum_weights(_tree.right_of(i)) 
      + _tree[i].first;
    return _tree[i].second;
  }

  Real total_weight;
  complete_tree<IntType, std::pair<Real, Real> > _tree;

};

}
}
#endif