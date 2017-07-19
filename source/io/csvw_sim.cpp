#include "csvw_sim.hpp"

#include <string>
using namespace std;



csvw_sim::csvw_sim(const std::string& pcfFileName, const RATETYPE& pcfTimeInterval,
        const RATETYPE& pcfTimeStart, const RATETYPE& pcfTimeEnd,
        const bool& pcfTimeColumn, const unsigned int& pcfCellTotal,
        const unsigned int& pcfCellStart, const unsigned int& pcfCellEnd,
        const specie_vec& pcfSpecieOption, Observable *pnObl) :
    csvw(pcfFileName, true, "\n# This file can be used as a template for "
            "user-created/modified analysis inputs in the context of this "
            "particular model for these particular command-line arguments.\n"),
    Observer(pnObl,pcfCellStart,pcfCellEnd,pcfTimeStart,pcfTimeEnd), icSpecieOption(pcfSpecieOption),
    ilTime(pcfTimeStart), ilCell(pcfCellStart), icTimeInterval(pcfTimeInterval), icTimeColumn(pcfTimeColumn),
    icTimeStart(pcfTimeStart), icTimeEnd(pcfTimeEnd),
    icCellTotal(pcfCellTotal), icCellStart(pcfCellStart), icCellEnd(pcfCellEnd)
{
    csvw::add_div("# The row after next MUST remain in the file in order for it "
            "to be parsable by the CSV reader. They indicate the following:\n"
            "cell-total, anlys-intvl, time-start, time-end, time-col, "
            "cell-start, cell-end, specie-option,\n");
    csvw::add_data(icCellTotal);
    csvw::add_data(icTimeInterval);
    csvw::add_data(icTimeStart);
    csvw::add_data(icTimeEnd);
    csvw::add_data(icTimeColumn);
    csvw::add_data(icCellStart);
    csvw::add_data(icCellEnd);
    for (unsigned int i=0; i<NUM_SPECIES; i++)
    {
        bool written = false;
        for (const specie_id& lcfID : icSpecieOption)
        {
            if ((specie_id) i == lcfID)
            {
                csvw::add_data(1);
                written = true;
                break;
            }
        }

        if (!written)
        {
            csvw::add_data(0);
        }
    }
    csvw::add_div("\n\n");


    if (icTimeColumn)
    {
        csvw::add_div("Time,");
    }
    for (const specie_id& lcfID : icSpecieOption)
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
    for (int c=min; c<max; c++){
        if (icTimeColumn)
        {
            csvw::add_data(ilTime);
        }

        for (const specie_id& lcfID : icSpecieOption)
        {
            csvw::add_data(pfStart.getCon(lcfID));
        }
        csvw::add_div("\n");
        pfStart.advance();
    }

    if (icCellTotal > 1)
    {
        csvw::add_div("\n");
    }
    ilTime += icTimeInterval;
}


