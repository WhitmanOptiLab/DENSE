#include "csvr.hpp"

#include "color.hpp"

#include <cfloat> // For FLT_MAX as an internal error code
#include <iostream>
using namespace std;



csvr::csvr(const std::string& pcfFileName)
{
    // Close any previously open file
    if (iFile.is_open())
        iFile.close();
    
    // Open new file
    iFile.open(pcfFileName);
    
    // Check if open successful
    if (!iFile.is_open())
        cout << color::set(color::RED) << "CSV file input failed. CSV file \'" << pcfFileName << "\' not found or open." << color::clear() << endl;
}


csvr::~csvr()
{
    iFile.close();
}


const bool csvr::is_open() const
{
    return iFile.is_open() ? true : false;
}


bool csvr::get_next(RATETYPE* pnRate)
{
    // Only bother if open
    if (iFile.is_open())
    {
        // tParam data from file to be "pushed" to pfRate
        string tParam;
        
        // For error reporting
        unsigned int lLine = 1;
        
        char c = iFile.get();
        while(iFile.good())
        {
            if (c == '#') // Check if in comment
            {
                // Skip comment line
                iFile.ignore(unsigned(-1), '\n');
                lLine++;
            }
            else if (c != ' ' && c != '\t') // Parse only if not whitespace except for \n
            {
                // Ignore non-numeric and non data seperator characters
                if (!((c >= '0' && c <= '9') || c == '.' || c == ',' || c == '\n'))
                {
                    iFile.ignore(unsigned(-1), ',');
                }
                else
                {
                    // If hit data seperator, add data to respective array
                    // '\n' is there in case there is no comma after last item
                    if (c == ',' || c == '\n')
                    {
                        if (tParam.length() > 0) // Only push if tParam contains something
                        {
                            RATETYPE tRate = FLT_MAX;
                            
                            try
                            {
                                tRate = stold(tParam);
                            }
                            catch(exception ex) // For catching stold() errors
                            {
                                cout << color::set(color::RED) << "CSV parsing failed. Invalid data contained at line " << lLine << "." << color::clear() << endl;
                            }
                            
                            // success
                            if (tRate != FLT_MAX)
                            {
                                pnRate != 0 ? *pnRate = tRate : 0;
                                return true;
                            }
                            else // error
                            {
                                return false;
                            }
                        }
                    }
                    else // Parse if it is numbers or decimal
                    {
                        tParam += c;
                    }
                }
            }
            
            // increment line counter
            if (c == '\n')
            {
                lLine++;
            }
            
            // get next char in file
            c = iFile.get();
        }
        
        // End of file
        return false;
    }
    else // if failed to open current_ifstream
    {
        cout << color::set(color::RED) << "CSV parsing failed. No CSV file found/open." << color::clear() << endl;
        return false;
    }
}
