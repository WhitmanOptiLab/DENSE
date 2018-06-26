#include "base.hpp"
#include "sim/base.hpp"

Analysis::Analysis (
  specie_vec const& species_vector,
  unsigned min_cell, unsigned max_cell,
  Real start_time, Real end_time
) :
  observed_species_{species_vector},
  start_time{start_time},
  end_time{end_time},
  min{min_cell},
  max{max_cell}
{
}

void Analysis::when_updated_by(Observable & observable) {
  if (observable.t < start_time || observable.t >= end_time) return;
  auto & begin = *observable.context;
  begin.set(min);
  update(begin);
}

void Analysis::when_unsubscribed_from(Observable & observable) {
  finalize();
}
