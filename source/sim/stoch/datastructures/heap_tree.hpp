#ifndef DENSE_SIM_HEAP_TREE
#define DENSE_SIM_HEAP_TREE
#include "completetree.hpp"
#include "heap.hpp"
#include "utility/numerics.hpp"
#include <limits>
#include <utility>
#include <random>

namespace dense {
namespace stochastic {


//Class to randomly select an index where each index's probability of being 
//  selected is weighted by a given vector.  
template <typename T, typename Compare = std::less<T>>
class heap_tree : public complete_tree<int, T>,  
                  public heap<heap_tree<T, Compare>, int> {
 public:
  using BaseTree = complete_tree<int, T>;
  using Heap = heap<heap_tree<T, Compare>, int>;
  friend Heap;
  using PosType = typename BaseTree::position_type;
  static PosType left_of(PosType i) { return BaseTree::left_of(i);}
  static PosType right_of(PosType i) { return BaseTree::right_of(i);}
  static PosType parent_of(PosType i) { return BaseTree::parent_of(i);}

  //Weights can be of any type, but most be convertable to Real values
  heap_tree() = delete;
  heap_tree(PosType p, Compare compare = Compare{}) : BaseTree(p), Heap(), _compare{compare} {};

  heap_tree(std::vector<T>& weights) :
    BaseTree(weights.size()),
    Heap()
  {
    for (auto w : weights) {
      BaseTree::add_entry(w);
      Heap::update_position(BaseTree::last());
    }
  }
 private:

  void swapWithChild(PosType parent, PosType child) {
    std::swap(BaseTree::value_of(parent), BaseTree::value_of(child));
  }

  void swap(PosType i, PosType j) {
    std::swap(BaseTree::value_of(i), BaseTree::value_of(j));
  }

  bool less(PosType a, PosType b) const {
    return _compare(BaseTree::value_of(a), BaseTree::value_of(b));
  }

  Compare _compare{};

  PosType id_of(PosType p) { return p; }

};

}
}
#endif
