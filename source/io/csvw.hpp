#ifndef IO_CSVW_HPP
#define IO_CSVW_HPP

#include "utility/numerics.hpp"
#include "utility/common_utils.hpp"

#include <fstream>
#include <string>
#include <memory>
#include <iostream>


class csvw
{
public:
    /**
     *  CSVWriter ctor
     *
     *  parameters
     *      pcfFileName - file name including ".csv" of the desired CSV file
     *
    */
    csvw(std::string const& pcfFileName);

    /**
     *  Add Data
     *  Add Data Divider (Separator)
     *
     *  usage
     *      For adding Real data and custom data separators to file
     *      add_data() automatically adds a "," between individual pieces of data
     *
     *  parameters
     *      pcfRate - the Real value to be written to file
     *      pcfDiv - the string to be written to file, probably
     *        for separating sections of data
     *
    */
    void add_data(Real const& pcfRate);
    void add_div(std::string const& pcfDiv);

    template <typename T>
    csvw & operator<< (T const& value) {
      stream() << value;
      return *this;
    }

    csvw(csvw&&) noexcept = default;
    csvw& operator=(csvw&&) noexcept = default;


    csvw(std::unique_ptr<std::ostream> stream)
    : stream_{std::move(stream)}
    , is_owner_{true} {
    }

    csvw(std::ostream& stream)
    : stream_{&stream}
    , is_owner_{false} {
    }

    ~csvw() {
      if (!is_owner_) stream_.release();
    }

    std::ostream& stream() {
      return *stream_;
    }

    std::ostream const& stream() const {
      return *stream_;
    }

    bool is_owner() const {
      return is_owner_;
    }

  protected:

    csvw()
    : csvw(std14::make_unique<std::ofstream>()) {
    }

  private:

    std::unique_ptr<std::ostream> stream_;
    bool is_owner_;

};

#endif
