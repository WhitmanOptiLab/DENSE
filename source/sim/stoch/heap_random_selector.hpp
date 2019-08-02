#ifndef DENSE_SIM_HEAP_TREE
#define DENSE_SIM_HEAP_TREE
#include "completetree.hpp"
#include "weightsum_tree.hpp"
#include "heap.hpp"
#include "utility/numerics.hpp"
#include <functional>
#include <limits>
#include <random>
#include <tuple>
#include <utility>
#include <iostream>

namespace dense {
namespace stochastic {


//Class to randomly select an index where each index's probability of being 
//  selected is weighted by a given vector.  
template <class IntType = int, size_t precision = std::numeric_limits<Real>::digits>
class heap_random_selector : protected complete_tree<IntType, std::tuple<IntType, Real, Real> >,  
                             public heap<heap_random_selector<IntType, precision>, IntType>,
                             protected weightsum_tree<heap_random_selector<IntType, precision>, IntType, IntType, precision> {
 public:
  using BaseTree = complete_tree<IntType, std::tuple<IntType, Real, Real> >;
  using Heap = heap<heap_random_selector<IntType, precision>, IntType>;
  using WeightSum = weightsum_tree<heap_random_selector<IntType, precision>, IntType, IntType, precision>;
  friend Heap;
  friend WeightSum;
  using PosType = typename BaseTree::position_type;

  heap_random_selector() = delete;
  heap_random_selector(PosType p) : BaseTree(p), Heap() {};

  bool empty() { return BaseTree::empty(); }

  template<typename T>
  heap_random_selector(std::vector<T>& weights) :
    BaseTree(weights.size()),
    Heap()
  {
    for (int i = 0; i < weights.size(); i++) {
      auto w = weights[i];
      BaseTree::add_entry(std::tuple<IntType, Real, Real>(i, w, 0.0));
      Heap::update_position(BaseTree::last());
    }
    WeightSum::compute_weights();
  }

  //Methods of BaseTree we want to make available
  std::tuple<IntType, Real, Real> top() {
    return BaseTree::top();
  }

  //Methods of WeightSum we want to make available
  template<class URNG>
  IntType operator()(URNG& g) { 
    return WeightSum::operator()(g);
  }

 private:
  void remove_last_entry() {
    auto last = BaseTree::value_of(BaseTree::last());
    WeightSum::update_weight(BaseTree::last(), 0);
    BaseTree::remove_last_entry();
  }

  void swap_with_child(PosType parent, PosType child) {
    std::swap(id_of(parent), id_of(child));
    WeightSum::swap_with_child(parent, child);
  }

  Real& weight_of(PosType p) {
    return std::get<1>(BaseTree::value_of(p));
  }
  Real& weightsum_of(PosType p) {
    return std::get<2>(BaseTree::value_of(p));
  }

  IntType& id_of(PosType p) { return std::get<0>(BaseTree::value_of(p)); }

  void swap(PosType i, PosType j) {
    std::swap(id_of(i), id_of(j));
    WeightSum::swap(i, j);
  }

  bool less(PosType a, PosType b) const {
    return std::greater<Real>()(std::get<1>(BaseTree::value_of(a)), std::get<1>(BaseTree::value_of(b)));
  }


};

}
}
#endif
