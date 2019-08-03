#ifndef IND_PRI_Q
#define IND_PRI_Q

#include <utility>
#include <vector>
#include <functional>
#include <type_traits>

#include "completetree.hpp"
#include "heap.hpp"
#include "indexed_collection.hpp"

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
        I>::type >,

    //... and the indexed collection mix-in
    public indexed_collection<  indexed_priority_queue<I, T, Compare>, 
      I, 
      typename std::conditional< std::is_enum<I>::value, 
        typename std::underlying_type< 
          typename std::conditional<std::is_enum<I>::value, I, ignore>::type>::type,
        I>::type,
      T>
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
      using Index = indexed_collection<indexed_priority_queue<index_type, mapped_type, Compare>, index_type, node_type, mapped_type>;

      friend Heap;
      friend Index;
   
      indexed_priority_queue() = delete;

      indexed_priority_queue(I max_size, Compare compare = Compare{}) :
        BaseTree(max_size),
        Heap(),
        Index(max_size),
        _compare{compare} {}

      indexed_priority_queue(indexed_priority_queue const&) = default;

      indexed_priority_queue(indexed_priority_queue &&) = default;

      indexed_priority_queue& operator=(indexed_priority_queue const&) = default;

      indexed_priority_queue& operator=(indexed_priority_queue &&) = default;

      ~indexed_priority_queue() = default;

      void push(value_type value) {
        auto node = Index::node_for_index(value.first);
        if (node == BaseTree::null_node()) {
          Heap::push(value);
        } else {
          Heap::update(node, value);
        }
      }

      void pop() { 
        auto index_removed = id_of(BaseTree::root());
        Heap::pop(); 
        Index::dissociate(index_removed);
      }

      mapped_type const& operator[](index_type i) const {
        return BaseTree::value_of(Index::node_for_index(i)).second;
      }

      const_iterator find(index_type i) const {
        auto node = Index::node_for_index(i);
        return node == BaseTree::null_node() ? BaseTree::end() : BaseTree::iterator_for(node);
      }

      mapped_type const& at(index_type i) const {
        auto node = Index::node_for_index(i);
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
      void add_entry(value_type v) {
        BaseTree::add_entry(v);
        auto newp = BaseTree::last();
        Index::associate(id_of(newp), newp);
      }

      indexed_priority_queue const& const_this() const {
        return static_cast<indexed_priority_queue const&>(*this);
      }

      bool less(node_type a, node_type b) const {
        return _compare(BaseTree::value_of(a).second, BaseTree::value_of(b).second);
      }

      void swap(node_type a, node_type b) {
        std::swap(BaseTree::value_of(a).second, BaseTree::value_of(b).second);
        Index::swap(a, b);
      }

      void swap_with_child(node_type a, node_type b) {
        swap(a, b);
      }

      index_type& id_of(node_type p) { return BaseTree::value_of(p).first; }

      mapped_compare _compare{};
  };
}
}

#endif
