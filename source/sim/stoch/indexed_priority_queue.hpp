#ifndef IND_PRI_Q
#define IND_PRI_Q

#include <utility>
#include <vector>
#include <functional>
#include <type_traits>

#include "completetree.hpp"
#include "heap.hpp"

namespace dense {
namespace stochastic {

namespace {
  enum class ignore{};
}

  template <
    typename I,
    typename T,
    typename Compare = std::less<T>
  >
  class indexed_priority_queue : 
    //Indexed priority queue extends a complete tree...
    public complete_tree<
      typename std::conditional< std::is_enum<I>::value, 
        typename std::underlying_type< 
          typename std::conditional<std::is_enum<I>::value, I, ignore>::type>::type,
        I>::type,
      std::pair<I, T> >,

    //... using the heap mix-in
    protected heap< indexed_priority_queue<I, T, Compare>, 
      typename std::conditional< std::is_enum<I>::value, 
        typename std::underlying_type< 
          typename std::conditional<std::is_enum<I>::value, I, ignore>::type>::type,
        I>::type >
  {

    private:

      template <typename E>
      using underlying_if_enum = typename std::conditional<
        std::is_enum<E>::value,
        typename std::underlying_type<
          typename std::conditional<std::is_enum<E>::value, E, ignore>::type
        >::type,
        E
      >::type;

    public:

      using size_type = std::ptrdiff_t;
      using index_type = I;
      using node_type = underlying_if_enum<index_type>;
      using mapped_type = T;
      using value_type = std::pair<index_type, mapped_type>;
      using mapped_compare = Compare;
      using iterator = value_type*;
      using const_iterator = value_type const*;
      using reference = value_type&;
      using const_reference = value_type const&;
      using BaseTree = complete_tree<node_type, value_type>;
      using Heap = heap<indexed_priority_queue<index_type, mapped_type, Compare>, node_type>;

      friend Heap;
   
      indexed_priority_queue() = delete;

      indexed_priority_queue(I max_size, Compare compare = Compare{}) :
        BaseTree(max_size),
        Heap(),
        _compare{compare},
        _node_for_index(max_size, max_size) {}

      indexed_priority_queue(indexed_priority_queue const&) = default;

      indexed_priority_queue(indexed_priority_queue &&) = default;

      indexed_priority_queue& operator=(indexed_priority_queue const&) = default;

      indexed_priority_queue& operator=(indexed_priority_queue &&) = default;

      ~indexed_priority_queue() = default;

      size_type max_size() const { return BaseTree::max_size(); }
      size_type size() const { return BaseTree::size(); }
      bool empty() const { return BaseTree::empty() == 0; }

      void push(value_type value) {
        auto& node = _node_for_index[value.first];
        if (node == BaseTree::null_node()) {
          Heap::push(value);
        } else {
          Heap::update(node, value);
        }
      }

      void pop() { Heap::pop(); }

      const_iterator begin() const { return BaseTree::iterator_for(BaseTree::root()); }
      const_iterator end() const { return BaseTree::end(); }
      const_reference top() const { return top(); }

      mapped_type const& operator[](index_type i) const {
        return BaseTree::value_of(_node_for_index[i]).second;
      }

      const_iterator find(index_type i) const {
        auto node = _node_for_index[i];
        return node == BaseTree::null_node() ? end() : BaseTree::iterator_for(node);
      }

      mapped_type const& at(index_type i) const {
        auto node = _node_for_index[i];
        if (node == BaseTree::null_node()) {
          throw std::out_of_range("Index out of range");
        }
        return BaseTree::value_of(node).second;
      }

      template <typename... Args>
      void emplace(Args&&... args) {
        push(value_type(std::forward<Args>(args)...));
      }

    private:

      indexed_priority_queue const& const_this() const {
        return static_cast<indexed_priority_queue const&>(*this);
      }

      bool less(node_type a, node_type b) const {
        return _compare(BaseTree::value_of(a).second, BaseTree::value_of(b).second);
      }

      void swap(node_type a, node_type b) {
        std::swap(BaseTree::value_of(a), BaseTree::value_of(b));
        _node_for_index[id_of(a)] = a;
        _node_for_index[id_of(b)] = b;
      }

      void swap_with_child(node_type a, node_type b) {
        swap(a, b);
      }

      index_type& id_of(node_type p) { return BaseTree::value_of(p).first; }

      mapped_compare _compare{};

      std::vector<node_type> _node_for_index;
  };

}
}

#endif
