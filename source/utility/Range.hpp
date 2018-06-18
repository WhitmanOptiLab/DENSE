#ifndef DENSE_RANGE_HPP
#define DENSE_RANGE_HPP

/// Released under the MIT license: https://varzea.mit-license.org/
template <typename T>
class Range {

  public:

    constexpr Range(T begin, T end) : begin_{begin}, end_{end} {};

    class Iterator {

      public:

        constexpr Iterator(T const& value) : value_{static_cast<decltype(value_)>(value)} {}

        Iterator & operator++ () { ++value_; return *this; }
        Iterator & operator-- () { --value_; return *this; }

        Iterator operator++ (signed) { return { value_++ }; }
        Iterator operator-- (signed) { return { value_-- }; }

        constexpr T operator* () const {
          return static_cast<T>(value_);
        }
        constexpr bool operator== (Iterator const& other) const {
          return value_ == other.value_;
        }
        constexpr bool operator!= (Iterator const& other) const {
          return !operator==(other);
        }

      private:

        typename std::conditional<std::is_enum<T>::value,
          typename std::underlying_type<T>::type, T
        >::type value_;
    };

    constexpr Iterator begin() const {
      return { begin_ };
    }

    constexpr Iterator end() const {
      return { end_ };
    }

  private:

    T begin_, end_;
};

template <typename T>
constexpr Range<T> make_range (T end) {
  return { T{}, end };
}

template <typename T>
constexpr Range<T> make_range (T begin, T end) {
  return { begin, end };
}

#endif
