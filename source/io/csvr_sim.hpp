#ifndef IO_CSVR_SIM_HPP
#define IO_CSVR_SIM_HPP

#include "csvr.hpp"
#include "sim/base.hpp"
#include "core/observable.hpp"
#include "core/specie.hpp"

#include <map>
#include <vector>

namespace dense {

class CSV_Streamed_Simulation : public csvr, public Simulation // <- order important here; csvr must be initialized before Simulation
{
  public:

    CSV_Streamed_Simulation(std::string const& pcfFileName, std::vector<Species> const& pcfSpecieVec);

    CSV_Streamed_Simulation(CSV_Streamed_Simulation&&) = default;
    CSV_Streamed_Simulation& operator=(CSV_Streamed_Simulation&&) = default;

    int getCellStart();
    int getCellEnd();

    Real get_concentration(dense::Natural cell, specie_id species) const override final {
      return iRate.at(cell).at(species);
    }

    Real get_concentration(dense::Natural cell, specie_id species, dense::Natural) const override final {
      return get_concentration(cell, species);
    }

    [[noreturn]] Real calculate_neighbor_average(dense::Natural, specie_id, dense::Natural) const override final {
      throw std::logic_error("Neighbor average not implemented for csvr_sim");
    }

    using Simulation::simulate_for;

    void simulate_for(Real duration) override final;

private:
    // Required for csvr_sim
    std::vector<Species> iSpecieVec;
    bool iTimeCol;
    int iCellStart, iCellEnd;
      std::vector<std::map<specie_id, Real>> iRate;
};

}

#endif
