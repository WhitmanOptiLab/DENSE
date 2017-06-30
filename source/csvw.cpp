#include "csvw.hpp"
#include "color.hpp"

#include <cfloat> // For FLT_MAX as an internal error code
#include <iostream>
using namespace std;



csvw::csvw(const std::string& pcfFileName, const bool& pcfWriteDoc, const string& pcfChildDoc)
{
    // Close any previously open file
    if (iFile.is_open())
        iFile.close();
    
    // Open new file
    iFile.open(pcfFileName);
    
    // Check if open successful
    if (!iFile.is_open())
    {
        cout << color::set(color::RED) << "CSV file output failed. CSV file \'" <<
            pcfFileName << "\' unable to be written to." << color::clear() << endl;
    }

    if (pcfWriteDoc)
    {
        // Write parent documentation
        iFile << "# CSV Specification\n";
        iFile << "#   Ignored by the file readers are:\n";
        iFile << "#     (1) blank rows and all other whitespace\n";
        iFile << "#     (2) comment rows which always begin with a \'#\'\n";
        iFile << "#     (3) blank cells such as \"A,B,,C,D\"\n";
        iFile << "#     (4) any cell that does not begin with a numerical or "
            "decimal place character*\n";
        iFile << "#   *Such cells act as either column headers or other important "
            "notes. These are provided for the user's convenience and can be "
            "modified if the user sees fit.\n";
        iFile << "#   It is futile to add/remove/modify the column headers with the "
            "expectation of changing the program's behavior. Data must be entered in "
            "the default order or must match the ordering of a respective command "
            "line argument.\n";
        iFile << "# For best results, DELETE ALL COMMENTS before loading this file"
            "into any Excel-like program.\n";

        // Write child documentation
        iFile << pcfChildDoc << endl;
    }
}


csvw::~csvw()
{
    iFile.close();
}


void csvw::add_div(const string& pcfDiv)
{
    iFile << pcfDiv;
}


void csvw::add_data(const RATETYPE& pcfRate)
{
    iFile << pcfRate;
    add_div(",");
}
