#include "csvw_sim.hpp"

#include <string>


csvw_sim::csvw_sim(std::string const& pcfFileName, RATETYPE const& pcfTimeInterval,
        RATETYPE const& pcfTimeStart, RATETYPE const& pcfTimeEnd,
        bool const& pcfTimeColumn, const unsigned int& pcfCellTotal,
        const unsigned int& pcfCellStart, const unsigned int& pcfCellEnd,
        specie_vec const& pcfSpecieOption, Observable & observable) :
    csvw(pcfFileName, true, "\n# This file can be used as a template for "
            "user-created/modified analysis inputs in the context of this "
            "particular model for these particular command-line arguments.\n"),
    Observer(), icSpecieOption(pcfSpecieOption),
    ilTime(pcfTimeStart), ilCell(pcfCellStart), icTimeInterval(pcfTimeInterval), icTimeColumn(pcfTimeColumn),
    icTimeStart(pcfTimeStart), icTimeEnd(pcfTimeEnd),
    icCellTotal(pcfCellTotal), icCellStart(pcfCellStart), icCellEnd(pcfCellEnd)
{
    subscribe_to(observable);
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
    for (unsigned int i = 0; i < NUM_SPECIES; i++)
    {
        bool written = false;
        for (specie_id const& lcfID : icSpecieOption)
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
    for (specie_id const& lcfID : icSpecieOption)
    {
        csvw::add_div(specie_str[lcfID] + ",");
    }
    csvw::add_div("\n");
}


csvw_sim::~csvw_sim()
{
}

void csvw_sim::when_updated_by(Observable & observable) {
  if (observable.t < icTimeStart || observable.t >= icTimeEnd) return;
  ContextBase & begin = *observable.context;
  begin.set(icCellStart);
  update(begin);
}

void csvw_sim::update(ContextBase& pfStart)
{
    for (int c = icCellStart; c < icCellEnd; ++c) {
        if (icTimeColumn)
        {
            csvw::add_data(ilTime);
        }

        for (specie_id const& lcfID : icSpecieOption)
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
