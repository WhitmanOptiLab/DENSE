#include "csvr_sim.hpp"

#include <iostream>

namespace dense {

CSV_Streamed_Simulation::CSV_Streamed_Simulation(std::string const& pcfFileName) :
    csvr(pcfFileName), Simulation()
{
    cell_count() = csvr::next<dense::Natural>();
    age_by(Minutes{ csvr::next<Real>() });
    iTimeCol = csvr::next<dense::Natural>() > 0;
    csvr::get_next(&iCellStart);
    csvr::get_next(&iCellEnd);

    // like a bitwise & of pcfSpecieVec and what exists in the file
    for (dense::Natural i = 0; i < NUM_SPECIES; i++)
    {
        Natural t;
        csvr::get_next(&t);
        if (t > 0)
        {
            iSpecieVec.push_back((specie_id) i);
        }
    }
}

int CSV_Streamed_Simulation::getCellStart()
{
    return iCellStart;
}


int CSV_Streamed_Simulation::getCellEnd()
{
    return iCellEnd;
}

/*
void csvr_sim::run()
{
    dense::Natural lCell = 0, lSpcVec = 0;
    iRate.reserve(NUM_SPECIES);
    Real hRate;

    // Skip first column
    t = iTimeCol ? csvr::next<dense::Natural>() : t + analysis_gran;

    while (csvr::get_next(&hRate)) {
        // Parse cells and push back maps of rows
        if (static_cast<std::size_t>(lCell) >= iRate.size()) {
          iRate.emplace_back();
        }
        iRate[lCell][iSpecieVec[lSpcVec]] = hRate;

        // Finished parsing row
        if (static_cast<std::size_t>(++lSpcVec) >= iSpecieVec.size())
        {
            lSpcVec = 0;

            // Skip first column
            t = iTimeCol ? csvr::next<dense::Natural>() : t + analysis_gran;

            // Finished parsing one time step
            if (++lCell >= _cells_total) {
                lCell = 0;
                notify();
                iRate.clear();
            }
        }
    }

    finalize();
}*/

Minutes CSV_Streamed_Simulation::age_by (Minutes duration) {
  Minutes stopping_time = age() + duration;
  while (age() < stopping_time) {
    iRate.resize(cell_count());
    for (dense::Natural cell = iCellStart; cell < iCellEnd; ++cell) {
      age_by((iTimeCol ? Minutes{ csvr::next<Real>() } : stopping_time) - age());
      for (auto & species : iSpecieVec) {
        iRate[cell][species] = csvr::next<Real>();
      }
    }
    iRate.clear();
  }
  return age();
}

}
