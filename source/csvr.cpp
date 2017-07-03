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
        cout << color::set(color::RED) << "CSV file input failed. CSV file \'" <<
            pcfFileName << "\' not found or open." << color::clear() << endl;
}


csvr::~csvr()
{
    iFile.close();
}


bool csvr::is_open() const
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
            // Check if in comment
            if (c == '#')
            {
                // Skip comment line
                iFile.ignore(unsigned(-1), '\n');
                lLine++;
            }
            // Parse only if not whitespace except for \n
            else if (c != ' ' && c != '\t')
            {
                // Ignore non-numeric, non data seperator, non e, -, and + characters
                if (!((c >= '0' && c <= '9') || c == '.' || c == ',' || c == '\n' ||
                        c == 'e' || c == '-' || c == '+'))
                {
                    iFile.ignore(unsigned(-1), ',');
                }
                else
                {
                    // If hit data seperator, add data to respective array
                    // '\n' is there in case there is no comma after last item
                    if (c == ',' || c == '\n')
                    {
                        // Only push if tParam contains something
                        if (tParam.length() > 0)
                        {
                            RATETYPE tRate = FLT_MAX;
                            
                            // catch stold() errors
                            try
                            {
                                tRate = stold(tParam);
                            }
                            catch(exception ex)
                            {
                                cout << color::set(color::RED) <<
                                    "CSV parsing failed. Invalid data contained "
                                    "at line " << lLine << "." <<
                                    color::clear() << endl;
                            }
                            
                            // if no error caught
                            if (tRate != FLT_MAX)
                            {
                                pnRate != 0 ? *pnRate = tRate : 0;
                                return true;
                            }
                            // else error caught
                            else
                            {
                                return false;
                            }
                        }
                    }
                    // Append if it is numbers or decimal
                    else
                    {
                        tParam += c;
                    }
                }
            }
            
            // Increment line counter
            if (c == '\n')
            {
                lLine++;
            }
            
            // Get next char in file
            c = iFile.get();
        }
        
        // End of file
        return false;
    }
    // If failed to open current_ifstream
    else
    {
        cout << color::set(color::RED) << "CSV parsing failed. "
            "No CSV file found/open." << color::clear() << endl;
        return false;
    }
}
