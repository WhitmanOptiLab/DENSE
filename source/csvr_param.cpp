#include "csvr_param.hpp"

#include <iostream>
#include <string>
using namespace std;


csvr_param::csvr_param(const string& pcfFileName) :
    csvr(pcfFileName)
{
    csvr tCopy(pcfFileName);
    param_set tDummy;
    while (tCopy.get_next(tDummy))
    {
        iCount++;
    }
    iCount /= (NUM_CRITICAL_SPECIES+NUM_DELAY_REACTIONS+NUM_REACTIONS-3);
    cout << "[DEBUG:csvr_param.cpp] iSetCount " << iCount << endl;
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


bool csvr_param::get_next(param_set& pfParam)
{
    bool rLoadSuccess = false;
    
    RATETYPE hRate;
    unsigned int lParamIndex = 0;
    unsigned int lArrayNum = 0;
    
    while (csvr::get_next(hRate))
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
            if (lParamIndex > NUM_CRITICAL_SPECIES - 1)
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
        if (lArrayNum > 2)
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
