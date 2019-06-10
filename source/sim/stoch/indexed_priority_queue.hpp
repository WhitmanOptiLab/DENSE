#include <utility>
#include <vector>
#include <functional>
#include <type_traits>

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
        _max_size{static_cast<node_type>(max_size)},
        _compare{compare},
        _heap(_max_size),
        _map(_max_size, null_node()) {}

      indexed_priority_queue(indexed_priority_queue const&) = default;

      indexed_priority_queue(indexed_priority_queue &&) = default;

      indexed_priority_queue& operator=(indexed_priority_queue const&) = default;

      indexed_priority_queue& operator=(indexed_priority_queue &&) = default;

      ~indexed_priority_queue() = default;

      size_type max_size() const {
        return _max_size;
      }

      size_type size() const {
        return _size;
      }

      bool empty() const {
        return _size == 0;
      }

      void push(value_type value) {
        auto& node = node_at(value.first);
        if (node == null_node()) {
          ++_size;
          node = last();
        }
        value_of(node) = value;
        sift_up(node) || sift_down(node);
      }

      void pop() {
        if (empty()) return;
        auto node = root();
        auto i = value_of(node).first;
        if (node != last()) {
          swap(node, last());
        }
        --_size;
        sift_down(node);
        node_at(i) = null_node();
      }

      const_iterator begin() const {
        return iterator_for(root());
      }

      const_iterator end() const {
        return iterator_for(last()) + 1;
      }

      const_reference top() const {
        return value_of(root());
      }

      mapped_type const& operator[](index_type i) const {
        return value_of(node_at(i)).second;
      }

      const_iterator find(index_type i) const {
        auto node = node_at(i);
        return node == null_node() ? end() : iterator_for(node);
      }

      mapped_type const& at(index_type i) const {
        auto node = node_at(i);
        if (node == null_node()) {
          throw std::out_of_range("Index out of range");
        }
        return value_of(node).second;
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
        return _compare(value_of(a).second, value_of(b).second);
      }

      node_type const& node_at(index_type i) const {
        return _map[static_cast<node_type>(i)];
      }

      node_type& node_at(index_type i) {
        return const_cast<node_type &>(const_this().node_at(i));
      }

      const_iterator iterator_for(node_type node) const {
        return _heap.data() + node;
      }

      iterator iterator_for(node_type node) {
        return const_cast<iterator>(const_this().iterator_for(node));
      }

      const_reference value_of(node_type node) const {
        return *iterator_for(node);
      }

      reference value_of(node_type node) {
        return const_cast<reference>(const_this().value_of(node));
      }

      static constexpr node_type root() { return 0; }
      node_type last() const { return _size - 1; }

      node_type parent_of(node_type node) const {
        return ((node + 1) >> 1) - 1;
      }

      node_type left_of(node_type node) const {
        return (node << 1) + 1;
      }

      node_type right_of(node_type node) const {
        return (node + 1) << 1;
      }

      node_type min_child_of(node_type node) const {
        auto left = left_of(node);
        auto right = right_of(node);
        return (right < _size && less(right, left)) ? right : left;
      }

      void swap(node_type a, node_type b) {
        using std::swap;
        swap(value_of(a), value_of(b));
        node_at(value_of(a).first) = a;
        node_at(value_of(b).first) = b;
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
        while (left_of(node) < _size && less(min_child = min_child_of(node), node)) {
          swap(node, min_child);
          node = min_child;
        }
        return node != start;
      }

      node_type _max_size;

      mapped_compare _compare{};

      node_type null_node() const {
        return _max_size;
      }

      std::vector<value_type> _heap;

      std::vector<node_type> _map;

      node_type _size = 0;

  };

}
}