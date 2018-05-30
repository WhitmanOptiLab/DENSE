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




csvr_sim::csvr_sim(const std::string& pcfFileName, const specie_vec& pcfSpecieVec) :
    csvr(pcfFileName)
{
    csvr::get_next(&iCellTotal);
    csvr::get_next(&iAnlysIntvl);
    csvr::get_next(&iTimeStart);
    csvr::get_next(&iTimeEnd);
    {
        int tTimeCol;
        csvr::get_next(&tTimeCol);
        iTimeCol = (tTimeCol > 0);
    }
    csvr::get_next(&iCellStart);
    csvr::get_next(&iCellEnd);

    // like a bitwise & of pcfSpecieVec and what exists in the file
    for (int i=0, t; i<NUM_SPECIES; i++)
    {
        csvr::get_next(&t);
        if (t > 0)
        {
            for (int j=0; j<pcfSpecieVec.size(); j++)
            {
                if (pcfSpecieVec[j] == (specie_id) i)
                {
                    iSpecieVec.push_back((specie_id) i);
                    break;
                }
            }
        }
    }
}


csvr_sim::~csvr_sim()
{
}


int csvr_sim::getCellTotal()
{
    return iCellTotal;
}


RATETYPE csvr_sim::getAnlysIntvl()
{
    return iAnlysIntvl;
}


RATETYPE csvr_sim::getTimeStart()
{
    return iTimeStart;
}


RATETYPE csvr_sim::getTimeEnd()
{
    return iTimeEnd;
}


int csvr_sim::getCellStart()
{
    return iCellStart;
}


int csvr_sim::getCellEnd()
{
    return iCellEnd;
}


void csvr_sim::run()
{
    int lCell = 0, lSpcVec = 0;
    csvr_sim::sim_ct hSCT;
    RATETYPE hRate;

    // Skip first column
    if (iTimeCol) csvr::get_next();
    
    while (csvr::get_next(&hRate))
    {
        // Parse cells and push back maps of rows
        if (lCell >= hSCT.iRate.size())
        {
            hSCT.iRate.push_back(map<specie_id, RATETYPE>());
        }
        hSCT.iRate[lCell][iSpecieVec[lSpcVec]] = hRate;

        // Finished parsing row
        if (++lSpcVec >= iSpecieVec.size())
        {
            lSpcVec = 0;

            // Skip first column
            if (iTimeCol) csvr::get_next();

            // Finished parsing one time step
            if (++lCell >= iCellTotal)
            {
                lCell = 0;
                notify(hSCT);
                hSCT.iRate.clear();
            }
        }
    }

    finalize();
}
