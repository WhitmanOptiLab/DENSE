#include "csvw.hpp"
#include "utility/style.hpp"
#include "utility/common_utils.hpp"

using style::Color;

#include <cfloat> // For FLT_MAX as an internal error code
#include <iostream>

csvw::csvw(std::string const& pcfFileName)
: stream_{}, is_owner_{true}
{
    auto file { std14::make_unique<std::ofstream>(pcfFileName) };
    // Check if open successful
    if (!file->is_open()) {
        std::cout << style::apply(Color::red) << "CSV file output failed. CSV file \'" <<
            pcfFileName << "\' unable to be written to." << style::reset() << '\n';
    }

    stream_ = std::move(file);
}

void csvw::add_div(std::string const& string) {
  stream() << string;
}

void csvw::add_data(Real const& real) {
  stream() << real << ',';
}
