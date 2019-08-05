#ifndef MODIFIABLE_HEAP_RANDOM_SELECTOR
#define MODIFIABLE_HEAP_RANDOM_SELECTOR

#include <utility>
#include <vector>
#include <functional>
#include <type_traits>

#include "completetree.hpp"
#include "heap.hpp"
#include "indexed_collection.hpp"
#include "weightsum_tree.hpp"

namespace dense {
namespace stochastic {

namespace {
  enum class ignore{};
}

  template <
    typename I, size_t precision = std::numeric_limits<Real>::digits>
  >
  class fast_random_selector : 
    //Extends a complete tree...
    protected complete_tree<
      typename std::conditional< std::is_enum<I>::value, 
        typename std::underlying_type< 
          typename std::conditional<std::is_enum<I>::value, I, ignore>::type>::type,
        I>::type,
      std::tuple<I, Real, Real> >,

    //... using the heap mix-in
    protected heap< fast_random_selector<I, precision>, 
      typename std::conditional< std::is_enum<I>::value, 
        typename std::underlying_type< 
          typename std::conditional<std::is_enum<I>::value, I, ignore>::type>::type,
        I>::type >,

    //... and the weightsum tree mix-in
    protected weightsum_tree< fast_random_selector<I, precision>,
      typename std::conditional< std::is_enum<I>::value, 
        typename std::underlying_type< 
          typename std::conditional<std::is_enum<I>::value, I, ignore>::type>::type,
        I>::type,
      int,
      precision>,

    //... AND the indexed collection mix-in
    protected indexed_collection<  fast_random_selector<I, precision>, 
      I, 
      typename std::conditional< std::is_enum<I>::value, 
        typename std::underlying_type< 
          typename std::conditional<std::is_enum<I>::value, I, ignore>::type>::type,
        I>::type,
      std::tuple<index_type, Real, Real>>
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
      using value_type = std::tuple<index_type, Real, Real> >,
      using iterator = value_type*;
      using const_iterator = value_type const*;
      using reference = value_type&;
      using const_reference = value_type const&;
      using BaseTree = complete_tree<node_type, value_type>;
      using Heap = heap<fast_random_selector<index_type, precision>, node_type>;
      using WeightSum = weightsum_tree<fast_random_selector<index_type, precision>, index_type, node_type, precision>;
      using Index = indexed_collection<fast_random_selector<index_type, precision>, index_type, node_type, mapped_type>;

      friend Heap;
      friend Index;
      friend WeightSum;
   
      fast_random_selector() = delete;

      fast_random_selector(I max_size) :
        BaseTree(max_size),
        Heap(),
        WeightSum(),
        Index(max_size),
      {}

      template<typename T>
      fast_random_selector(std::vector<T>& weights) :
          BaseTree(weights.size()),
          Heap(),
          WeightSum(),
          Index(max_size),
      {
        for (int i = 0; i < weights.size(); i++) {
          auto w = weights[i];
          //Heap::push will add entries through the add_entry method, which 
          //  will create index associations
          Heap::push(std::tuple<index_type, Real, Real>(i, Real(w), 0.0));
        }
        WeightSum::compute_weights();
      }

      fast_random_selector(fast_random_selector const&) = default;

      fast_random_selector(fast_random_selector &&) = default;

      fast_random_selector& operator=(fast_random_selector const&) = default;

      fast_random_selector& operator=(fast_random_selector &&) = default;

      ~fast_random_selector() = default;
      
      
      //Methods of WeightSum we want to make available
      template<class URNG>
      index_type operator()(URNG& g) { 
        return id_of(WeightSum::operator()(g));
      }

      void update_weight(index_type i, Real new_weight) {
        auto node = Index::node_for_index(i);
        WeightSum::update_weight(Index::node_for_index(i), new_weight);
        Heap::update_position(node);
      }

      Real total_weight() { return _total_weight; }

    private:
      //Must call WeightSum::compute_weights() after this, before using random selection
      void add_entry(value_type v) {
        BaseTree::add_entry(v);
        auto newp = BaseTree::last();
        Index::associate(id_of(newp), newp);
      }

      void remove_last_entry() {
        auto last = BaseTree::value_of(BaseTree::last());
        WeightSum::update_weight(BaseTree::last(), 0);
        BaseTree::remove_last_entry();
      }

      fast_random_selector const& const_this() const {
        return static_cast<fast_random_selector const&>(*this);
      }

      //Max-heap property
      bool less(node_type a, node_type b) const {
        return BaseTree::value_of(a).second > BaseTree::value_of(b).second;
      }

      void swap(node_type a, node_type b) {
        std::swap(BaseTree::value_of(a).second, BaseTree::value_of(b).second);
        WeightSum::swap(a, b);
        Index::swap(a, b);
      }

      void swap_with_child(node_type a, node_type b) {
        Index::swap(a, b);
        WeightSum::swap_with_child(a, b);
      }

      index_type& id_of(node_type p) { return BaseTree::value_of(p).first; }

  };
}
}

#endif
