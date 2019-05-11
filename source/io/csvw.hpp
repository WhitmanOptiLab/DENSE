#ifndef IO_CSVW_HPP
#define IO_CSVW_HPP

#include "utility/numerics.hpp"

#include <fstream>
#include <string>


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
    csvw(std::string const& pcfFileName, bool const& pcfWriteDoc = true, std::string const& pcfChildDoc = "");

    /**
     *  Add Data
     *  Add Data Divider (Seperator)
     *
     *  usage
     *      For adding Real data and custom data seperators to file
     *      add_data() automatically adds a "," between individual pieces of data
     *
     *  parameters
     *      pcfRate - the Real value to be written to file
     *      pcfDiv - the string to be written to file, probably
     *        for seperating sections of data
     *
    */
    void add_data(Real const& pcfRate);
    void add_div(std::string const& pcfDiv);

    template <typename T>
    csvw & operator<< (T const& value) {
      iFile << value;
      return *this;
    }

private:
    // Output file of CSV
    std::ofstream iFile;
};

#endif
