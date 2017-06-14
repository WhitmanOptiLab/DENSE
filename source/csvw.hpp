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
    
    
    // TODO : Document
    void add_div(const std::string& pcfDiv);
    void add_data(const RATETYPE& pcfRate);

private:
    // Output file of CSV
    std::ofstream iFile;
};

#endif
