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
  Simulation * simulation = dynamic_cast<Simulation*>(&observable);
  if (simulation && (simulation->t < start_time || simulation->t >= end_time)) return;
  update({ simulation, min });
}

void Analysis::when_unsubscribed_from(Observable & observable) {
  finalize();
}
