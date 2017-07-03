#include "csvw_sim.hpp"

#include <string>
using namespace std;



csvw_sim::csvw_sim(const string& pcfFileName, const RATETYPE& pcfTimeInterval, const unsigned int& pcfCellTotal, const specie_vec& pcfSpecieVec, Observable *pnObl) :
    csvw(pcfFileName, true, "# This file can be used as a template for user-created/modified inputs in the context of this particular model using this particular \'-o | --specie-option\' setting.\n"), Observer(pnObl), oSpecieVec(pcfSpecieVec), iTimeCount(0), iTimeInterval(pcfTimeInterval), iCellTotal(pcfCellTotal)
{
    //csvw::add_div("Time, Cell, ");

    csvw::add_div("# Appropriate Command Line Arguments (Non-Comprehensive)\nDELETE THE NEXT LINE BEFORE LOADING THIS CSV INTO A PROGRAM LIKE EXCEL OR GOOGLE SPREADSHEETS!!!\n# --specie-option \"");
    for (const specie_id& lcfID : oSpecieVec)
    {
        csvw::add_div(specie_str[lcfID] + ",");
        // Having a comma after the last specie is no big deal
    }
    csvw::add_div("\" --cell-total " + to_string(pcfCellTotal) + " --anlys-intvl " + to_string(pcfTimeInterval) + "\n\n");

/*
    csvw::add_div("Time,");
    for (const specie_id& lcfID : oSpecieVec)
    {
        csvw::add_div(specie_str[lcfID] + ",");
    }
    csvw::add_div("\n");*/
}


csvw_sim::~csvw_sim()
{
}


void csvw_sim::finalize(ContextBase& pfStart)
{
}


void csvw_sim::update(ContextBase& pfStart)
{
    unsigned int lCell = 0;
  //  csvw::add_data(iTimeInterval*iTimeCount++);
   // RATETYPE averages[oSpecieVec.size()];

    while (pfStart.isValid())
    {
        csvw::add_div(to_string(iTimeCount*iTimeInterval)+", "+to_string(lCell++)+", ");
        for (const specie_id& lcfID : oSpecieVec)
        {
            csvw::add_data(pfStart.getCon(lcfID));
        }
        csvw::add_div("\n");
        pfStart.advance();

       /* for (int i=0; i<oSpecieVec.size(); i++)
        {
            averages[i] += pfStart.getCon(oSpecieVec[i]);
        }
        pfStart.advance();*/
    }

    csvw::add_div("\n");
    iTimeCount++;
    
/*
    for (int i=0; i<oSpecieVec.size(); i++)
    {
        csvw::add_data(averages[i]/iCellTotal);
    }
    csvw::add_div("\n");
*/
}


