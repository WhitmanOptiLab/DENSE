#ifndef RANDOM_SELECTOR
#define RANDOM_SELECTOR
#include "completetree.hpp"
#include "weightsum_tree.hpp"
#include "utility/numerics.hpp"
#include <limits>
#include <utility>
#include <random>

namespace dense {
namespace stochastic {


//Class to randomly select an index where each index's probability of being 
//  selected is weighted by a given vector.  
template <class IntType = int, size_t precision = std::numeric_limits<Real>::digits>
class nonuniform_int_distribution : protected complete_tree<IntType, std::pair<Real, Real> >,  
                                    public weightsum_tree<nonuniform_int_distribution<IntType, precision>, IntType, precision> {
 public:
  using BaseTree = complete_tree<IntType, std::pair<Real, Real> >;
  using WeightSum = weightsum_tree<nonuniform_int_distribution<IntType, precision>, IntType, precision>;
  friend WeightSum;
  using PosType = typename BaseTree::position_type;
  static PosType left_of(PosType i) { return BaseTree::left_of(i);}
  static PosType right_of(PosType i) { return BaseTree::right_of(i);}
  static PosType parent_of(PosType i) { return BaseTree::parent_of(i);}

  //Weights can be of any type, but most be convertable to Real values
  nonuniform_int_distribution() = delete;
  nonuniform_int_distribution(PosType p) : BaseTree(p), WeightSum(*this) {};

  template<typename T>
  nonuniform_int_distribution(std::vector<T>& weights) :
    BaseTree(weights.size()),
    WeightSum(*this)
  {
    for (auto w : weights) {
      BaseTree::emplace_entry(w, 0.0);
    }
    WeightSum::compute_weights();
  }

  Real& weight_of(PosType p) {
    return BaseTree::value_of(p).first;
  }
  Real& weightsum_of(PosType p) {
    return BaseTree::value_of(p).second;
  }

  PosType id_of(PosType p) { return p; }

};

}
}
#endif
