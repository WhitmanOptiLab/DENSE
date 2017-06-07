#include "color.hpp"
#include "param_set.hpp"

#include <exception>
#include <fstream>
#include <iostream>
using namespace std;



// See documentation in header
CSVReader param_set::isCSV;

// See documentation in header
unsigned int param_set::isTotal = 0;
unsigned int param_set::isRemaining = 0;



// See usage documentation in header
bool param_set::open_file(const string& pcfFileName)
{
    isCSV.open(pcfFileName);
    
    // Create an identical ifstream to isCSV for iteration. This is because for some reason I had trouble resetting the stream back to the beginning using seekg(0) after the stream made its way to the end. By creating a copy, I don't have to mess with isCSV's ifstream position.
    ifstream tIFSCopy(pcfFileName);
    
    // If open successful, count data sets and return true
    // This cannot be a function of CSVReader because the definition of a "set" could be different (i.e. a single line or a group of lines) depending on the particular need
    if (tIFSCopy.is_open())
    {
        // Data lines counter
        isTotal = isRemaining = 0;
        
        // The first char of each line should give away whether it's a data line or not. Count these data lines.
        char c = tIFSCopy.get();
        while(tIFSCopy.good())
        {
            if ((c >= '0' && c <= '9') || c == '.') // Only increment if it's a number or decimal
            {
                isTotal++;
                isRemaining++;
            }
            
            // Skip entire or rest of line
            tIFSCopy.ignore(unsigned(-1), '\n');
            c = tIFSCopy.get();
            
            // "Trim" all excess whitespace
            while (c == ' ' || c == '\t' || c == '\n')
            {
                tIFSCopy.ignore(unsigned(-1), c);
                c = tIFSCopy.get();
            }
        }
        
        tIFSCopy.close();
        return true;
    }
    else
    {
        cout << color::set(color::RED) << "Failed to open \'" << pcfFileName << "\'." << color::clear() << endl;
        tIFSCopy.close();
        return false;
    }
}



// See usage documentation in header
void param_set::close_file()
{
    isCSV.close();
    isTotal = isRemaining = 0;
}



// See usage documentation in header
unsigned int param_set::get_set_total()
{
    return isTotal;
}



// See usage documentation in header
unsigned int param_set::get_set_remaining()
{
    return isRemaining;
}



// See usage documentation in header
param_set param_set::load_next_set()
{
    param_set rSet;
    load_next_set(rSet);
    return rSet;
}



// See usage documentation in header
bool param_set::load_next_set(param_set &pfLoadTo)
{
    bool rLoadSuccess = false;
    
    RATETYPE hRate;
    unsigned int lParamIndex = 0;
    unsigned int lArrayNum = 0;
    
    while (isCSV.nextCSVCell(hRate))
    {
        RATETYPE *nToArray = nullptr;
        
        switch (lArrayNum)
        {
            // This ordering must match that of gen_csv.cpp
        case 0:
            nToArray = pfLoadTo._rates_base;
            if (lParamIndex > NUM_REACTIONS - 1)
            {
                lParamIndex = 0;
                lArrayNum++;
            }
            else
            {
                break;
            }
        case 1:
            nToArray = pfLoadTo._delay_sets;
            if (lParamIndex > NUM_DELAY_REACTIONS - 1)
            {
                lParamIndex = 0;
                lArrayNum++;
            }
            else
            {
                break;
            }
        case 2:
            nToArray = pfLoadTo._critical_values;
            if (lParamIndex >= NUM_CRITICAL_SPECIES - 1)
            {
                // Mind the >=
                // Break one index early so that we don't reach the next set
                lArrayNum++;
            }
            else
            {
                break;
            }
        default:
            nToArray = nullptr;
        }
        
        
        // If nToArray is set to something, try adding data to nToArray
        if (nToArray != nullptr)
        {
            nToArray[lParamIndex++] = hRate;
            !rLoadSuccess ? rLoadSuccess = true : 0;
        }
        
        
        // Now break if reached end
        if (lArrayNum > 2)
        {
            break;
        }
    }
    
    // Decrement isRemaining if successful
    rLoadSuccess ? isRemaining-- : 0;
    
    return rLoadSuccess;
}

