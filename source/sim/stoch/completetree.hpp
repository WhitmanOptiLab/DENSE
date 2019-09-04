#ifndef COMPLETE_TREE
#define COMPLETE_TREE

#include <utility>
#include <vector>
#include <functional>
#include <type_traits>

namespace dense {
namespace stochastic {



  /* This complete tree class is meant to be subclassed to build various 
   * data structures.  Because most of those data structures require 
   * updates when values are modified, only const access to elements is 
   * granted publicly, while there are protected methods for non-const 
   * element access.
   */
  template <
    typename P,
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
      using position_type = P;
      using entry_type = T;
      using iterator = entry_type*;
      using const_iterator = entry_type const*;
      using reference = entry_type&;
      using const_reference = entry_type const&;
   
      complete_tree() = delete;

      complete_tree(position_type max_size) :
        _max_size{max_size},
        _tree(_max_size) {}

      complete_tree(complete_tree const&) = default;

      complete_tree(complete_tree &&) = default;

      complete_tree& operator=(complete_tree const&) = default;

      complete_tree& operator=(complete_tree &&) = default;

      ~complete_tree() = default;

      //Tree property methods
      size_type max_size() const {
        return static_cast<position_type>(_max_size);
      }

      size_type size() const {
        return _size;
      }

      bool empty() const {
        return _size == 0;
      }

      //Tree modification methods
      //
    protected:
      void add_entry(entry_type entry) {
        _tree.at(last()+1) = entry;
        _size++;
      }

      template <typename... Args>
      void emplace_entry(Args&&... args) {
        add_entry(entry_type(std::forward<Args>(args)...));
      }

      void remove_last_entry() {
        if (empty()) return;
        _size--;
      }

      //Iterator methods
    public:
      const_iterator begin() const {
        return iterator_for(root());
      }

      const_iterator end() const {
        return iterator_for(last()) + 1;
      }

      const_iterator iterator_for(position_type node) const {
        return _tree.data() + node;
      }

    protected:
      iterator iterator_for(position_type node) {
        return const_cast<iterator>(const_this().iterator_for(node));
      }
      
      iterator begin() {
        return iterator_for(root());
      }

      iterator end() {
        return iterator_for(last()) + 1;
      }

    public:
      //Position methods
      static constexpr position_type root() { return 0; }
      position_type last() const { return _size - 1; }
      position_type null_node() const {
        return _max_size;
      }

      static position_type parent_of(position_type node) {
        return ((node + 1) >> 1) - 1;
      }

      static position_type left_of(position_type node) {
        return (node << 1) + 1;
      }

      static position_type right_of(position_type node) {
        return (node + 1) << 1;
      }

      //Element access
      const_reference top() const {
        return value_of(root());
      }

      const_reference operator[](position_type i) const {
        return _tree[i];
      }

      const_reference at(position_type i) const {
        if (i >= _size) {
          throw std::out_of_range("Index out of range");
        }
        return _tree[i];
      }


      complete_tree const& const_this() const {
        return static_cast<complete_tree const&>(*this);
      }

      const_reference value_of(position_type node) const {
        return *iterator_for(node);
      }

    protected:
      reference value_of(position_type node) {
        return const_cast<reference>(const_this().value_of(node));
      }

      reference operator[](position_type i) {
        return _tree[i];
      }

      reference at(position_type i) {
        if (i >= _size) {
          throw std::out_of_range("Index out of range");
        }
        return _tree[i];
      }
      
    private:
      position_type _max_size;
      std::vector<entry_type> _tree;
      position_type _size = 0;

  };

}
}
#endif
