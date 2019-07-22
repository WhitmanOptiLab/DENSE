#include <utility>
#include <vector>
#include <functional>
#include <type_traits>

#include "completetree.hpp"

namespace dense {
namespace stochastic {

  template <
    typename I,
    typename T,
    typename Compare = std::less<T>
  >
  class indexed_priority_queue {

    private:

      enum class ignore {};

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
   
      indexed_priority_queue() = delete;

      indexed_priority_queue(I max_size, Compare compare = Compare{}) :
        _compare{compare},
        _heap(max_size),
        _map(max_size, null_node()) {}

      indexed_priority_queue(indexed_priority_queue const&) = default;

      indexed_priority_queue(indexed_priority_queue &&) = default;

      indexed_priority_queue& operator=(indexed_priority_queue const&) = default;

      indexed_priority_queue& operator=(indexed_priority_queue &&) = default;

      ~indexed_priority_queue() = default;

      size_type max_size() const { return _heap.max_size(); }
      size_type size() const { return _heap.size(); }
      bool empty() const { return _heap.empty() == 0; }

      void push(value_type value) {
        auto& node = node_at(value.first);
        if (node == null_node()) {
          _heap.push(value);
          node = _heap.last();
        } else {
          _heap.value_of(node) = value;
        }
        sift_up(node) || sift_down(node);
      }

      void pop() {
        if (empty()) return;
        auto node = root();
        auto i = _heap.value_of(node).first;
        if (node != _heap.last()) {
          swap(node, _heap.last());
        }
        _heap.pop();
        sift_down(node);
        node_at(i) = null_node();
      }

      const_iterator begin() const { return _heap.iterator_for(root()); }
      const_iterator end() const { return _heap.end(); }
      const_reference top() const { return _heap.top(); }

      mapped_type const& operator[](index_type i) const {
        return _heap.value_of(node_at(i)).second;
      }

      const_iterator find(index_type i) const {
        auto node = node_at(i);
        return node == null_node() ? end() : _heap.iterator_for(node);
      }

      mapped_type const& at(index_type i) const {
        auto node = node_at(i);
        if (node == null_node()) {
          throw std::out_of_range("Index out of range");
        }
        return _heap.value_of(node).second;
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
        return _compare(_heap.value_of(a).second, _heap.value_of(b).second);
      }

      node_type const& node_at(index_type i) const {
        return _map[static_cast<node_type>(i)];
      }

      node_type& node_at(index_type i) {
        return const_cast<node_type &>(const_this().node_at(i));
      }

      static constexpr node_type root() { return decltype(_heap)::root(); }

      node_type parent_of(node_type node) const { return _heap.parent_of(node); }
      node_type left_of(node_type node) const { return _heap.left_of(node); }
      node_type right_of(node_type node) const { return _heap.right_of(node); }

      node_type min_child_of(node_type node) const {
        auto left = left_of(node);
        auto right = right_of(node);
        return (right < _heap.size() && less(right, left)) ? right : left;
      }

      void swap(node_type a, node_type b) {
        using std::swap;
        swap(_heap.value_of(a), _heap.value_of(b));
        node_at(_heap.value_of(a).first) = a;
        node_at(_heap.value_of(b).first) = b;
      }

      bool sift_up(node_type node) {
        node_type start = node, parent;
        while (node != root() && less(node, parent = parent_of(node))) {
          swap(node, parent);
          node = parent;
        }
        return node != start;
      }

      bool sift_down(node_type node) {
        node_type start = node, min_child;
        while (left_of(node) < _heap.size() && less(min_child = min_child_of(node), node)) {
          swap(node, min_child);
          node = min_child;
        }
        return node != start;
      }

      mapped_compare _compare{};

      node_type null_node() const { return _heap.null_node(); }

      complete_tree<node_type, value_type> _heap;

      std::vector<node_type> _map;
  };

}
}
