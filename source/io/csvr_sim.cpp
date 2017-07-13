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


void csvr_sim::sim_ct::set(int c)
{
    iIter = c;
}




csvr_sim::csvr_sim(const string& pcfFileName, const specie_vec& pcfSpecieVec) :
    csvr(pcfFileName), icSpecieVec(pcfSpecieVec)
{
    // csvr::get_next(&blah); to get all configs
}


csvr_sim::~csvr_sim()
{
}


void csvr_sim::run()
{
    int lCell = 0, lSpcVec = 0;
    csvr_sim::sim_ct hSCT;
    RATETYPE hRate;

    // Skip initial columns here if necessary
    // csvr::get_next();
    
    while (csvr::get_next(&hRate))
    {
        if (lCell >= hSCT.iRate.size())
        {
            hSCT.iRate.push_back(map<specie_id, RATETYPE>());
        }
        hSCT.iRate[lCell][icSpecieVec[lSpcVec]] = hRate;

        if (++lSpcVec >= icSpecieVec.size())
        {
            lSpcVec = 0;

            // Skip initial columns here if necessary
            // csvr::get_next();

            if (++lCell >= /*icCellTotal TODO*/20)
            {
                lCell = 0;
                notify(hSCT);
                hSCT.iRate.clear();
            }
        }
    }

    // A blank, dummy sim_ct for the sake of finalize()
    finalize();
}
