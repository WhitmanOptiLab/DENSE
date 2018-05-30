#include "base.hpp"

Analysis::Analysis (
  Observable * log,
  specie_vec const& species_vector,
  csvw * csv_out,
  int min, int max,
  Real start_time, Real end_time
) :
  Observer(log, min, max, start_time, end_time),
  time{0},
  ucSpecieOption(species_vector),
  csv_out(csv_out) {
};
