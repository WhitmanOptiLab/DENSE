#include "csvw_sim.hpp"

#include <string>
using namespace std;



csvw_sim::csvw_sim(const string& pcfFileName, const RATETYPE& pcfTimeInterval,
        const unsigned int& pcfCellTotal, const specie_vec& pcfSpecieVec,
        Observable *pnObl) :
    csvw(pcfFileName, true, "# This file can be used as a template for"
            "user-created/modified inputs in the context of this particular"
            "model using this particular \'-o | --specie-option\' setting.\n"
            "# For best results, DELETE ALL COMMENTS before loading this file"
            "into any Excel-like program.\n"),
    Observer(pnObl), oSpecieVec(pcfSpecieVec), iTimeCount(0),
    icTimeInterval(pcfTimeInterval), icCellTotal(pcfCellTotal)
{
    csvw::add_div("# Respective Command Line Arguments"
            "(Non-Comprehensive)\n# --specie-option \"");
    for (const specie_id& lcfID : oSpecieVec)
    {
        csvw::add_div(specie_str[lcfID] + ",");
        // Having a comma after the last specie is no big deal
    }
    csvw::add_div("\" --cell-total " + to_string(icCellTotal) +
            " --anlys-intvl " + to_string(icTimeInterval) + "\n\n");


    /*
    // Was used to output averages over all cells.
    // CAUTION: Such output is not valid for csvr_sim.
    csvw::add_div("Time,");
    for (const specie_id& lcfID : oSpecieVec)
    {
        csvw::add_div(specie_str[lcfID] + ",");
    }
    csvw::add_div("\n");
    */
}


csvw_sim::~csvw_sim()
{
}


void csvw_sim::finalize(ContextBase& pfStart)
{
}


void csvw_sim::update(ContextBase& pfStart)
{
    /*
    // Was used to output averages over all cells.
    csvw::add_data(icTimeInterval*iTimeCount++);
    RATETYPE averages[oSpecieVec.size()];
    */


    while (pfStart.isValid())
    {
        for (const specie_id& lcfID : oSpecieVec)
        {
            csvw::add_data(pfStart.getCon(lcfID));
        }
        csvw::add_div("\n");
        pfStart.advance();
        break;


        /*
        // Was used to output averages over all cells.
        for (int i=0; i<oSpecieVec.size(); i++)
        {
            averages[i] += pfStart.getCon(oSpecieVec[i]);
        }
        pfStart.advance();
        */
    }

    if (icCellTotal > 1)
        csvw::add_div("\n");
   

    /*
    // Was used to output averages over all cells.
    for (int i=0; i<oSpecieVec.size(); i++)
    {
        csvw::add_data(averages[i]/icCellTotal);
    }
    csvw::add_div("\n");
    */
}


