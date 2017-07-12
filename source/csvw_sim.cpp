#include "csvw_sim.hpp"

#include <string>
using namespace std;



csvw_sim::csvw_sim(const string& pcfFileName, const RATETYPE& pcfTimeInterval,
        const unsigned int& pcfCellTotal, const specie_vec& pcfSpecieVec,
        Observable *pnObl, int min_cell, int max_cell, RATETYPE startT, RATETYPE endT) :
    csvw(pcfFileName, true, "# This file can be used as a template for"
            "user-created/modified inputs in the context of this particular"
            "model for these particular species in the \'-o | "
            "--specie-option\' setting.\n"),
    Observer(pnObl, min_cell, max_cell, startT, endT), oSpecieVec(pcfSpecieVec), iTimeCount(0),
    icTimeInterval(pcfTimeInterval), icCellTotal(pcfCellTotal)
{
    csvw::add_div("# Hint: Use the column headers to help derive the appropriate "
            "\'-o | --specie-option\' setting.\n");
    csvw::add_div("# A couple more necessary but not comprehensive command "
            "line arguments:\n");
    csvw::add_div("--cell-total " + to_string(icCellTotal) +
            " --anlys-intvl " + to_string(icTimeInterval) + "\n\n");


    // Was used to output averages over all cells.
    // CAUTION: Such output is not valid for csvr_sim.
    //csvw::add_div("Time,");
    
    
    for (const specie_id& lcfID : oSpecieVec)
    {
        csvw::add_div(specie_str[lcfID] + ",");
    }
    csvw::add_div("\n");
}


csvw_sim::~csvw_sim()
{
}


void csvw_sim::finalize()
{
}


void csvw_sim::update(ContextBase& pfStart)
{
    // Was used to output averages over all cells.
    /*
    csvw::add_data(icTimeInterval*iTimeCount++);
    RATETYPE averages[oSpecieVec.size()];
    */


    for (int c=min; c<max; c++)
    {
        for (const specie_id& lcfID : oSpecieVec)
        {
            csvw::add_data(pfStart.getCon(lcfID));
        }
        //csvw::add_div("\n");
        pfStart.advance();

        // Was used to output averages over all cells.
        /*
        for (int i=0; i<oSpecieVec.size(); i++)
        {
            averages[i] += pfStart.getCon(oSpecieVec[i]);
        }
        pfStart.advance();
        */
    }

    if (icCellTotal > 1)
    {
        csvw::add_div("\n");
    }
   

    // Was used to output averages over all cells.
    /*
    for (int i=0; i<oSpecieVec.size(); i++)
    {
        csvw::add_data(averages[i]/icCellTotal);
    }
    csvw::add_div("\n");
    */
}


