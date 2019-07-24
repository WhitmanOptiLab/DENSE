#ifndef COMPLETE_TREE
#define COMPLETE_TREE

#include <utility>
#include <vector>
#include <functional>
#include <type_traits>

namespace dense {
namespace stochastic {

  template <
    typename I,
    typename T
  >
  class complete_tree {

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
      using entry_type = T;
      using iterator = entry_type*;
      using const_iterator = entry_type const*;
      using reference = entry_type&;
      using const_reference = entry_type const&;
   
      complete_tree() = delete;

      complete_tree(I max_size) :
        _max_size{max_size},
        _tree(_max_size) {}

      complete_tree(complete_tree const&) = default;

      complete_tree(complete_tree &&) = default;

      complete_tree& operator=(complete_tree const&) = default;

      complete_tree& operator=(complete_tree &&) = default;

      ~complete_tree() = default;

      size_type max_size() const {
        return _max_size;
      }

      size_type size() const {
        return _size;
      }

      bool empty() const {
        return _size == 0;
      }

      void push(entry_type entry) {
        _tree.at(last()+1) = entry;
        _size++;
      }

      void pop() {
        if (empty()) return;
        _size--;
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

      const_reference operator[](index_type i) const {
        return _tree[i];
      }

      reference operator[](index_type i) {
        return _tree[i];
      }

      const_reference at(index_type i) const {
        if (i >= _size) {
          throw std::out_of_range("Index out of range");
        }
        return _tree[i];
      }

      reference at(index_type i) {
        if (i >= _size) {
          throw std::out_of_range("Index out of range");
        }
        return _tree[i];
      }

      template <typename... Args>
      void emplace(Args&&... args) {
        push(entry_type(std::forward<Args>(args)...));
      }

      complete_tree const& const_this() const {
        return static_cast<complete_tree const&>(*this);
      }

      const_iterator iterator_for(index_type node) const {
        return _tree.data() + node;
      }

      iterator iterator_for(index_type node) {
        return const_cast<iterator>(const_this().iterator_for(node));
      }

      const_reference value_of(index_type node) const {
        return *iterator_for(node);
      }

      reference value_of(index_type node) {
        return const_cast<reference>(const_this().value_of(node));
      }

      static constexpr index_type root() { return 0; }
      index_type last() const { return _size - 1; }

      index_type parent_of(index_type node) const {
        return ((node + 1) >> 1) - 1;
      }

      index_type left_of(index_type node) const {
        return (node << 1) + 1;
      }

      index_type right_of(index_type node) const {
        return (node + 1) << 1;
      }

      index_type _max_size;

      index_type null_node() const {
        return _max_size;
      }

      std::vector<entry_type> _tree;

      index_type _size = 0;

  };

}
}
#endif