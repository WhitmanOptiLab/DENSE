#include "csvr_param.hpp"

#include <iostream>
#include <string>



csvr_param::csvr_param(std::string const& pcfFileName) :
    csvr(pcfFileName), iCount(0), iRemain(0)
{
    csvr tCopy(pcfFileName);
    while (tCopy.get_next())
    {
        iCount++;
    }
    iCount /= (NUM_CRITICAL_SPECIES + NUM_DELAY_REACTIONS + NUM_REACTIONS);
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
    
    while (csvr::get_next(&hRate))
    {
        RATETYPE *nToArray = pfLoadTo.getArray();
        
        // If nToArray is set to something, try adding data to nToArray
        if (nToArray != nullptr)
        {
            nToArray[lParamIndex++] = hRate;
            !rLoadSuccess ? rLoadSuccess = true : 0;
        }

        // Now break if reached end
        if (lParamIndex == NUM_CRITICAL_SPECIES + NUM_REACTIONS + NUM_DELAY_REACTIONS)
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
