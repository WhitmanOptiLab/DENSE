#include "csvr_sim.hpp"

#include <iostream>

namespace dense {

CSV_Streamed_Simulation::CSV_Streamed_Simulation(std::string const& pcfFileName, std::vector<Species> const& pcfSpecieVec) :
    csvr(pcfFileName), Simulation()
{
    _cells_total = csvr::next<dense::Natural>();
    age_ = csvr::next<Real>();
    iTimeCol = csvr::next<dense::Natural>() > 0;
    csvr::get_next(&iCellStart);
    csvr::get_next(&iCellEnd);

    // like a bitwise & of pcfSpecieVec and what exists in the file
    for (dense::Natural i = 0; i < NUM_SPECIES; i++)
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

void CSV_Streamed_Simulation::simulate_for (Real duration) {
  Real stopping_time = age_ + duration;
  while (age_ < stopping_time) {
    iRate.resize(_cells_total);
    for (dense::Natural cell = iCellStart; cell < iCellEnd; ++cell) {
      age_ = iTimeCol ? csvr::next<Real>() : stopping_time;
      for (auto & species : iSpecieVec) {
        iRate[cell][species] = csvr::next<Real>();
      }
    }
    iRate.clear();
  }
}

}
