#include "csvw_sim.hpp"

#include <string>
using namespace std;



csvw_sim::csvw_sim(const string& pcfFileName, const RATETYPE& pcfTimeInterval, const specie_vec& pcfSpecieVec, Observable *pnObl) :
    csvw(pcfFileName, true, "# This file can be used as a template for user-created data inputs under this particular model using this particular \'-o | --specie-option\' setting.\n"), iTimeInterval(pcfTimeInterval), Observer(pnObl), oSpecieVec(pcfSpecieVec), iTimeCount(1)
{
    csvw::add_div("Time, Cell, ");
    for (const specie_id& lcfID : oSpecieVec)
    {
        csvw::add_div(specie_str[lcfID] + ", ");
    }
    csvw::add_div("\n");
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
    while (pfStart.isValid())
    {
        csvw::add_div(to_string(iTimeCount*iTimeInterval)+", "+to_string(lCell++)+", ");
        for (const specie_id& lcfID : oSpecieVec)
        {
            csvw::add_data(pfStart.getCon(lcfID));
        }
        csvw::add_div("\n");
        pfStart.advance();
    }

    iTimeCount++;
}


