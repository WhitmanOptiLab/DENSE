#ifndef INDEXED_COLLECTION
#define INDEXED_COLLECTION

#include <utility>
#include <vector>
#include <functional>
#include <type_traits>

#include "completetree.hpp"
#include "heap.hpp"

namespace dense {
namespace stochastic {

//An indexed collection is meant to be a mix-in of a complete tree.
//It requires that tree entries store an ID of some kind, and keeps an 
//  up-to-date map of where entries corresponding to particular user-provided 
//  indexes can be found.
template <
  class Tree, 
  typename IndexType,
  typename PosType,
  typename MappedType
>
class indexed_collection {
  public:
    indexed_collection() = delete;
    indexed_collection(IndexType max_size) :
      _node_for_index(max_size, max_size) {}

    //Element access
    const MappedType& operator[](IndexType i) const {
      return _tree().value_of(_node_for_index[i]);
    }

    MappedType const& at(IndexType i) const {
      auto node = _node_for_index[i];
      if (node == _tree().null_node()) {
        throw std::out_of_range("Index out of range");
      }
      return _tree().value_of(node);
    }

  protected:
    PosType node_for_index(IndexType i) const {
      return _node_for_index[i];
    }

    MappedType& operator[](IndexType i) {
      return _tree().value_of(_node_for_index[i]);
    }


    void associate(IndexType i, PosType p) {
      _node_for_index[i] = p;
    }

    void dissociate(IndexType i) {
      _node_for_index[i] = _tree().null_node();
    }

    void swap(PosType a, PosType b) {
      std::swap(_tree().id_of(a), _tree().id_of(b));
      associate(_tree().id_of(a), a);
      associate(_tree().id_of(b), b);
    }

    void swap_with_child(PosType a, PosType b) {
      swap(a, b);
    }

  private:
    std::vector<PosType> _node_for_index;
    Tree& _tree() { return *static_cast<Tree*>(this); }
    const Tree& _tree() const { return *static_cast<const Tree*>(this); }
};

}
}

#endif
