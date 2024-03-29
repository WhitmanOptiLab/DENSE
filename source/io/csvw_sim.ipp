#include "csvw_sim.hpp"

#include <string>
#include <limits>

namespace dense {

template <typename Simulation>
csvw_sim<Simulation>::csvw_sim(Natural cell_total,
  bool pcfTimeColumn,
  std::vector<Species> const& pcfSpecieOption) :
  Analysis<Simulation>(pcfSpecieOption, { 0, cell_total }),
  icTimeColumn(pcfTimeColumn)
{
}

template <typename Simulation>
void csvw_sim<Simulation>::update (Simulation& simulation, std::ostream& log) {
  if (print_header_) {

    log << "\n# This file can be used as a template for "
      "user-created/modified analysis inputs in the context of this "
      "particular model for these particular command-line arguments.\n";

    log << "# The row after next MUST remain in the file in order for it "
      "to be parsable by the CSV reader. They indicate the following:\n"
      "cell-total, time-start, time-col, "
      "cell-start, cell-end, specie-option,\n";
    log <<
      this->max << ',' <<
      simulation.age().count() << ',' <<
      icTimeColumn << ',' <<
      this->min << ',' << this->max << ',';

    for (unsigned int i = 0; i < NUM_SPECIES; i++) {
      bool written = false;
      for (specie_id const& lcfID : this->observed_species_) {
        if ((specie_id) i == lcfID) {
          log << 1.0 << ',';
            written = true;
            break;
            }
        }

      if (!written)
      {
        log << 0.0 << ',';
      }
    }
    log << "\n\n";


    if (icTimeColumn)
    {
      log << "Time,";
    }
    if (this->icCellColumn){
      log << "Cell,";
    }
    for (specie_id const& lcfID : this->observed_species_)
    {
      log << specie_str[lcfID] << ',';
    }
    log << '\n';
    print_header_ = false;
  }
    
  dense::Natural physical_cell = 0;
  bool once = true;
  for (int& virtual_cell : this->physical_cells_id) {
    if(virtual_cell >= 0){
      if (icTimeColumn && once) {
        log << "Time: " << simulation.age().count() << ',';
      } else if (icTimeColumn && !once) log << ',';
      if(this->icCellColumn){
        log << "Cell: " << virtual_cell << ",";
      }
      for (specie_id const& lcfID : this->observed_species_) {
        log << simulation.get_concentration(physical_cell, lcfID) << ',';
      }
      log << '\n';
      once = false;
    }
    physical_cell++;
  }

  //if simulating more than one cell, add a space between the time blocks
  if (this->max > this->min) {
    log << '\n';
  }
}

}
