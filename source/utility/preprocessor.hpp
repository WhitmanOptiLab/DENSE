#ifndef DENSE_UTILITY_PREPROCESSOR_HPP
#define DENSE_UTILITY_PREPROCESSOR_HPP

#include <iosfwd>

/// Yield a string literal representing the supplied arguments
///   as they were prior to macro expansion.
#define PREPROCESSOR_GET_TEXT(...) #__VA_ARGS__

/// Yield a string literal representing the supplied arguments
///   as they were after macro expansion.
#define PREPROCESSOR_GET_EXPANDED_TEXT(...) PREPROCESSOR_GET_TEXT(__VA_ARGS__)

namespace preprocessor {

  /// A compile-time view to an immutable range of characters.
  class Text {

    public:

      /// Random-access contiguous iterator type alias.
      using Iterator = char const*;

      /// Construct a text view to a fixed-length character array.
      template <std::ptrdiff_t size>
      constexpr Text(char const (& data)[size])
      : Text(data, data + size) {
      }

      /// Construct a text view to `size` characters, beginning at `data`.
      ///   Expects that `[data, data + size)` is a valid range of characters.
      constexpr Text(char const* first, char const* last)
      : first_{first}, last_{last} {
      }

      /// Get an iterator to the beginning of a text view.
      constexpr Iterator begin() const noexcept { return first_; }

      /// Get an iterator to the end of a text view.
      constexpr Iterator end() const noexcept { return last_; }

      /// Get a pointer to the first character of a text view.
      constexpr char const* data () const noexcept { return begin(); }

      /// Get the size of a text view.
      constexpr std::ptrdiff_t size () const noexcept { return last_ - first_; }

      /// Get the character at the specified offset within a text view.
      ///   Expects 0 <= `index` < `size()`.
      constexpr char operator[] (std::ptrdiff_t index) const noexcept {
        return data()[index];
      }

      constexpr Iterator find (char) const noexcept;

      constexpr bool is_subset_of (Text const& other) const noexcept;

    private:

      char const* first_;
      char const* last_;

  };


  namespace {

    template <typename Forward_Iterator>
    constexpr bool equal(Forward_Iterator first, Forward_Iterator last, Forward_Iterator other_first) noexcept {
      return first == last || (*first == *other_first && equal(first + 1, last, other_first + 1));
    }

    template <typename Forward_Iterator, typename Value>
    constexpr Forward_Iterator find_(Forward_Iterator first, Forward_Iterator last, Value value) noexcept {
      return (first == last || *first == value) ? first : find_(first + 1, last, value);
    }

    template <typename Forward_Iterator>
    constexpr bool is_subset_of_(Forward_Iterator first, Forward_Iterator last, Text const& alphabet) noexcept {
      return first == last || (alphabet.find(*first) != alphabet.end() && is_subset_of_(first + 1, last, alphabet));
    }

  }

  constexpr Text::Iterator Text::find (char value) const noexcept {
    return find_(begin(), end(), value);
  }

  constexpr bool Text::is_subset_of(Text const& other) const noexcept {
    return is_subset_of_(begin(), end(), other);
  }

  /// Determine whether a text view is equal to another.
  ///   Two text views `a` and `b` are considered equal if they have the
  ///   same size, and `a[i] == b[i]` for all `i` in `[0, a.size())`.
  constexpr bool operator== (Text const& a, Text const& b) noexcept {
    return a.size() == b.size() && equal(a.begin(), a.end(), b.begin());
  }

  /// Insert the characters of a text view into a `std::ostream`.
  inline std::ostream & operator<< (std::ostream & out, Text const& text);

}

#endif
