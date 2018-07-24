#ifndef IO_CSVR_H
#define IO_CSVR_H

#include "utility/numerics.hpp"

#include <fstream>
#include <string>
#include <memory>


class csvr
{
public:
    /**
     *  CSVReader ctor
     *
     *  parameters
     *      pcfFileName - file name including ".csv" of the desired CSV file
     *      pcfSuppressWarning - enable/disable message saying file name does not exist
    */
    csvr(std::string const& file_name, bool suppress_file_not_found = false);

    csvr(csvr&&) = default;
    csvr& operator=(csvr&&) = default;

    bool has_stream() const;

    /**
     *  Get Next CSV Data Cell
     *
     *  usage
     *      Because this returns a bool, it can easily be used in a while loop
     *      Use this as a helper function inside other functions that load
     *        data from specially organized CSVs into classes such as param_set
     *        and datalogger
     *
     *  parameters
     *      pnRate - Real pointer you want the next data cell to be stored in
     *
     *  returns
     *      true - if there is a next cell and pfRate was successfully set to it
     *      false - if reached end of file or a parsing error
    */
    bool get_next(Real* pnRate);
    bool get_next(int* pnRate);
    bool skip_next();

    template <typename T>
    T next() {
      T result;
      if (!get_next(&result)) {
        throw std::out_of_range("Could not get next!");
      }
      return result;
    }

    static bool get_real(std::istream& in, Real* out);

private:
    // Input file of CSV
    std::unique_ptr<std::istream> iFile;
    unsigned int iLine;
};

#endif
