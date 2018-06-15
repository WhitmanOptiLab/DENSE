#include "csvr_sim.hpp"

#include <iostream>




csvr_sim::sim_ct::sim_ct() :
    iIter(0)
{
    iRate.reserve(NUM_SPECIES);
}


Real csvr_sim::sim_ct::getCon(specie_id sp) const
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




csvr_sim::csvr_sim(std::string const& pcfFileName, specie_vec const& pcfSpecieVec) :
    csvr(pcfFileName)
{
    int total;
    csvr::get_next(&total);
    iCellTotal = total;
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
    for (unsigned i = 0; i < NUM_SPECIES; i++)
    {
        int t;
        csvr::get_next(&t);
        if (t > 0)
        {
            for (std::size_t j = 0; j < pcfSpecieVec.size(); j++)
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


Real csvr_sim::getAnlysIntvl()
{
    return iAnlysIntvl;
}


Real csvr_sim::getTimeStart()
{
    return iTimeStart;
}


Real csvr_sim::getTimeEnd()
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
    unsigned lCell = 0, lSpcVec = 0;
    csvr_sim::sim_ct hSCT;
    Real hRate;

    // Skip first column
    if (iTimeCol) csvr::get_next();

    while (csvr::get_next(&hRate))
    {
        // Parse cells and push back maps of rows
        if (lCell >= hSCT.iRate.size())
        {
            hSCT.iRate.push_back(std::map<specie_id, Real>());
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
