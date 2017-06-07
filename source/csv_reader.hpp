#ifndef CSV_READ_H
#define CSV_READ_H

#include <fstream>
#include <string>

#include "reaction.hpp" // For "typedef float RATETYPE;"


class CSVReader
{
public:
    /**
     *  CSVReader ctor
     *  
     *  parameters
     *      pcfFileName - file name including ".csv" of the desired CSV file
     *      pcnFile - a pointer to the desired output file stream
    */
    CSVReader(const std::string& pcfFileName = "");
    
    
    // Like the ctor but can be used afterwards
    bool open(const std::string& pcfFileName);
    void close();
    
    
    /**
     *  Next CSV Data Cell
     *
     *  usage
     *      Because this returns a bool, it can easily be used in a while loop
     *      Use this as a helper function inside other functions that load data from specially organized CSVs into classes such as param_set and datalogger
     *  
     *  parameters
     *      pfRate - RATETYPE variable you want the next data cell to be stored in
     *      
     *  returns
     *      true - if there is a next cell and pfRate was successfully set to it
     *      false - if reached end of file or a parsing error
    */
    bool nextCSVCell(RATETYPE& pfRate);
    
    
private:
    // Input file of CSV
    std::ifstream iFile;
};

#endif
