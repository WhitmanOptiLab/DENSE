#ifndef IO_CSVR_H
#define IO_CSVR_H

#include "util/common_utils.hpp" // For "using RATETYPE = float;"

#include <fstream>
#include <string>


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
    csvr(std::string const& pcfFileName, bool const& pcfSuppressWarning = false);
    virtual ~csvr();

    bool is_open() const;

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
     *      pnRate - RATETYPE pointer you want the next data cell to be stored in
     *
     *  returns
     *      true - if there is a next cell and pfRate was successfully set to it
     *      false - if reached end of file or a parsing error
    */
    bool get_next(RATETYPE* pnRate);
    bool get_next(int* pnRate);
    bool get_next();


private:
    // Input file of CSV
    std::ifstream iFile;
    unsigned int iLine;
};

#endif
