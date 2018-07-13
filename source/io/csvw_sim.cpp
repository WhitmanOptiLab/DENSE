#include "csvw_sim.hpp"

#include <string>
#include <limits>

namespace dense {

csvw_sim::csvw_sim(std::string const& pcfFileName,
        bool pcfTimeColumn,
        specie_vec const& pcfSpecieOption, Simulation & observable) :
    csvw(pcfFileName, true, "\n# This file can be used as a template for "
            "user-created/modified analysis inputs in the context of this "
            "particular model for these particular command-line arguments.\n"),
    Observer(), icSpecieOption(pcfSpecieOption),
    icCellTotal(observable._cells_total), icCellEnd(icCellTotal),
    icTimeColumn(pcfTimeColumn)
{
    subscribe_to(observable);
    csvw::add_div("# The row after next MUST remain in the file in order for it "
            "to be parsable by the CSV reader. They indicate the following:\n"
            "cell-total, time-start, time-col, "
            "cell-start, cell-end, specie-option,\n");
    csvw::add_data(icCellTotal);
    csvw::add_data(observable.t);
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

void csvw_sim::when_updated_by(Simulation & simulation) {
  if (simulation.t < icTimeStart || simulation.t >= icTimeEnd) return;
  update({ &simulation, icCellStart });
}

void csvw_sim::update(dense::Context<> pfStart)
{
    for (unsigned c = icCellStart; c < icCellEnd; ++c) {
        if (icTimeColumn)
        {
            csvw::add_data(pfStart.time());
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
}

}
