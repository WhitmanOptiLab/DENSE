#include "csvr.hpp"
#include "util/color.hpp"

#include <cfloat> // For FLT_MAX as an internal error code
#include <cmath> // For round()
#include <iostream>
using namespace std;



csvr::csvr(std::string const& pcfFileName, bool const& pcfSuppressWarning) :
    iLine(1)
{
    // Close any previously open file
    if (iFile.is_open())
        iFile.close();
    
    // Open new file
    iFile.open(pcfFileName);
    
    // Check if open successful
    if (!iFile.is_open() && !pcfSuppressWarning)
        std::cout << color::set(color::RED) << "CSV file input failed. CSV file \'" <<
            pcfFileName << "\' not found or open." << color::clear() << '\n';
}


csvr::~csvr()
{
    iFile.close();
}


bool csvr::is_open() const
{
    return iFile.is_open() ? true : false;
}


bool csvr::get_next()
{
    RATETYPE tDummy;
    return csvr::get_next(&tDummy);
}


bool csvr::get_next(int* pnRate)
{
    RATETYPE tFloating;
    if (csvr::get_next(&tFloating))
    {
        if (pnRate)
        {
            *pnRate = (int) round(tFloating);
        }

        return true;
    }
    else
    {
        return false;
    }
}


bool csvr::get_next(RATETYPE* pnRate)
{
    // Only bother if open
    if (iFile.is_open())
    {
        // tParam data from file to be "pushed" to pfRate
        string tParam;
        
        char c = iFile.get();
        while(iFile.good())
        {
            // Check if in comment
            if (c == '#')
            {
                // Skip comment line
                iFile.ignore(unsigned(-1), '\n');
                tParam.clear();
                iLine++;
            }
            // Parse only if not whitespace except for \n
            else if (c != ' ' && c != '\t')
            {
                // Ignore non-numeric, non data seperator, non e, -, and + characters
                if (!( (c >= '0' && c <= '9') || c == '.' || c == ',' || c == '\n' ||
                        c == 'e' || c == '-' || c == '+') )
                {
                    iFile.ignore(unsigned(-1), ',');
                    tParam.clear();
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
                            char* tInvalidAt;
                            RATETYPE tRate = strtold(tParam.c_str(), &tInvalidAt);
                            
                            // If found invalid while parsing
                            if (*tInvalidAt)
                            {
                                std::cout << color::set(color::RED) <<
                                    "CSV parsing failed. Invalid data contained "
                                    "at line " << iLine << "." <<
                                    color::clear() << '\n';
                                return false;
                            }
                            // Else was success
                            else
                            {
                                pnRate != 0 ? *pnRate = tRate : 0;
                                return true;
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
                iLine++;
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
        std::cout << color::set(color::RED) << "CSV parsing failed. "
            "No CSV file found/open." << color::clear() << '\n';
        return false;
    }
}
