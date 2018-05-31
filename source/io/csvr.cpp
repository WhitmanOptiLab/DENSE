#include "csvr.hpp"
#include "util/color.hpp"

#include <cfloat> // For FLT_MAX as an internal error code
#include <cmath>
#include <iostream>


csvr::csvr(std::string const& file_name, bool suppress_file_not_found) :
    iLine(1), iFile(file_name)
{
    if (!iFile.is_open() && !suppress_file_not_found)
        std::cout << color::set(color::RED) << "CSV file input failed. CSV file \'" <<
            file_name << "\' not found or open." << color::clear() << '\n';
}

bool csvr::is_open() const {
    return iFile.is_open();
}

bool csvr::get_next() {
    return csvr::get_next(static_cast<RATETYPE *>(nullptr));
}

bool csvr::get_next(int* rate) {
  RATETYPE result;
  bool success = get_next(&result);
  if (success && rate) *rate = std::round(result);
  return success;
}

bool csvr::get_next(RATETYPE* pnRate) {
    // Only bother if open
    if (iFile.is_open())
    {
        // tParam data from file to be "pushed" to pfRate
        std::string tParam;

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
