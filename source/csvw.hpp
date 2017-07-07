#ifndef CSVW_H
#define CSVW_H

#include <fstream>
#include <string>

#include "reaction.hpp" // For "typedef float RATETYPE;"


class csvw
{
protected:
    /**
     *  CSVWriter ctor
     *  
     *  parameters
     *      pcfFileName - file name including ".csv" of the desired CSV file
     *      
    */
    csvw(const std::string& pcfFileName, const bool& pcfWriteDoc = true, const std::string& pcfChildDoc = "");
    virtual ~csvw();
    
    /**
     *  Add Data
     *  Add Data Divider (Seperator)
     *
     *  usage
     *      For adding RATETYPE data and custom data seperators to file
     *      add_data() automatically adds a "," between individual pieces of data
     *  
     *  parameters
     *      pcfRate - the RATETYPE value to be written to file
     *      pcfDiv - the string to be written to file, probably
     *        for seperating sections of data
     *      
    */
    void add_data(const RATETYPE& pcfRate);
    void add_div(const std::string& pcfDiv);

private:
    // Output file of CSV
    std::ofstream iFile;
};

#endif
