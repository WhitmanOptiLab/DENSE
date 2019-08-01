#ifndef WEIGHTSUM_TREE
#define WEIGHTSUM_TREE
#include "completetree.hpp"
#include "utility/numerics.hpp"
#include <limits>
#include <utility>
#include <random>

namespace dense {
namespace stochastic {


//Class to randomly select an index where each index's probability of being 
//  selected is weighted by a given vector.  
template <class Tree, class PosType, class IntType = int, size_t precision = std::numeric_limits<Real>::digits>
class weightsum_tree {
 public:
  //Weights can be of any type, but most be convertable to Real values
  weightsum_tree() = default;

  void compute_weights() {
    total_weight = sum_weights(_tree().root());
  }

  void update_weight(PosType i, Real new_weight) {
    Real weight_diff = new_weight - _tree().weight_of(i);
    while (i != _tree().root()) {
      _tree().weightsum_of(i) += weight_diff;
      i = Tree::parent_of(i);
    }
    _tree().weightsum_of(i) += weight_diff;
    total_weight += weight_diff;
  }

  template<class URNG>
  IntType operator()(URNG& g) {

    Real target = std::generate_canonical<Real, precision, URNG>(g)*total_weight;
    PosType node = _tree().root();
    //Loop until target random value is in between weight(left) and weight(left) + value(node)
    while(checked_weightsum(Tree::left_of(node)) > target ||
             _tree().weight_of(node) + checked_weightsum(Tree::left_of(node)) < target) {
      if (checked_weightsum(Tree::left_of(node)) > target) {
        node = Tree::left_of(node);
      } else {
        target -= _tree().weight_of(node) + _tree().weightsum_of(Tree::left_of(node));
        node = Tree::right_of(node);
      }
      //Should this ever happen?  No, but floating-point rounding means it's 
      //  theoretically possible, and it's better to reboot than crash at this point
      if (node > _tree().last()) {
        node = _tree().root();
        target = std::generate_canonical<Real, precision, URNG>(g)*total_weight;
      }
    }
    return _tree().id_of(node);
  }

 private:
  Tree& _tree() { return *static_cast<Tree*>(this); }
  Real checked_weightsum(PosType node) { return node > _tree().last() ? 0 : _tree().weightsum_of(node); }
  Real checked_weight(PosType node) { return node > _tree().last() ? 0 : _tree().weight_of(node); }
  Real sum_weights(PosType i) {
    if (i > _tree().last()) return 0.0f;
    _tree().weightsum_of(i) =
      sum_weights(Tree::left_of(i)) + sum_weights(Tree::right_of(i)) 
      + _tree().weight_of(i);
    return _tree().weightsum_of(i);
  }

  Real total_weight;
};

}
}
#endif