#include "csvr_sim.hpp"

#include <iostream>
using namespace std;



csvr_sim::sim_ct::sim_ct() :
    iIter(0)
{
    iRate.reserve(NUM_SPECIES);
}


RATETYPE csvr_sim::sim_ct::getCon(specie_id sp) const
{
    return iRate[iIter].at(sp);
}


void csvr_sim::sim_ct::advance()
{
    iIter++;
}


bool csvr_sim::sim_ct::isValid() const
{
    return iIter < iRate.size();
}


void csvr_sim::sim_ct::reset()
{
    iIter = 0;
}




csvr_sim::csvr_sim(const string& pcfFileName, const unsigned int& pcfCellTotal, const specie_vec& pcfSpecieVec) :
    csvr(pcfFileName), iSpecieVec(pcfSpecieVec), iCellTotal(pcfCellTotal)
{
}


csvr_sim::~csvr_sim()
{
}


void csvr_sim::run()
{
    int lCell = 0, lSpcVec = 0;
    csvr_sim::sim_ct hSCT;
    RATETYPE hRate;

    // Skip first two columns
    csvr::get_next();
    csvr::get_next();
    
    while (csvr::get_next(&hRate))
    {
        if (lCell >= hSCT.iRate.size())
        {
            hSCT.iRate.push_back(map<specie_id, RATETYPE>());
        }
        hSCT.iRate[lCell][iSpecieVec[lSpcVec]] = hRate;

        if (lSpcVec++ >= iSpecieVec.size())
        {
            lSpcVec = 0;

            // Skip first two columns
            csvr::get_next();

            if (++lCell >= iCellTotal)
            {
                lCell = 0;
                notify(hSCT);
                hSCT.iRate.clear();
            }
        }
    }
}
