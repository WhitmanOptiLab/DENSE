#include "csvr_param.hpp"

#include <iostream>
#include <string>
using namespace std;


csvr_param::csvr_param(const string& pcfFileName) :
    csvr(pcfFileName), iCount(0), iRemain(0)
{
    csvr tCopy(pcfFileName);
    while (tCopy.get_next())
    {
        iCount++;
    }
    iCount /= (NUM_CRITICAL_SPECIES+NUM_DELAY_REACTIONS+NUM_REACTIONS-3);
    iRemain = iCount;
}


const unsigned int& csvr_param::get_total() const
{
    return iCount;
}


const unsigned int& csvr_param::get_remain() const
{
    return iRemain;
}


param_set csvr_param::get_next()
{
    param_set rPS;
    get_next(rPS);
    return rPS;
}


bool csvr_param::get_next(param_set& pfLoadTo)
{
    bool rLoadSuccess = false;
    
    RATETYPE hRate;
    unsigned int lParamIndex = 0;
    unsigned int lArrayNum = 0;
    
    while (csvr::get_next(&hRate))
    {
        RATETYPE *nToArray = nullptr;
        
        switch (lArrayNum)
        {
            // This ordering must match that of gen_csv.cpp
        case 0:
            nToArray = pfLoadTo._rates_base;
            if (!(lParamIndex < NUM_REACTIONS))
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
            if (!(lParamIndex < NUM_DELAY_REACTIONS))
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
            if (!(lParamIndex < NUM_CRITICAL_SPECIES))
            {
                lParamIndex = 0;
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
        if (lArrayNum == 2 && lParamIndex == NUM_CRITICAL_SPECIES)
        {
            break;
        }
    }
    
    // Decrement isRemaining if successful
    rLoadSuccess ? iRemain-- : 0;
    
    return rLoadSuccess;
}


csvr_param::~csvr_param()
{

}
