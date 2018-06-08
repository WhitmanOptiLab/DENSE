#include "base.hpp"

Analysis::Analysis (
  Observable * log,
  specie_vec const& species_vector,
  csvw * csv_out,
  int min, int max,
  Real start_time, Real end_time
) :
  min{min},
  max{max},
  start_time{start_time},
  end_time{end_time},
  time{0},
  ucSpecieOption(species_vector),
  csv_out(csv_out) {
  subscribe_to(*log);
};

void Analysis::when_updated_by(Observable & observable) {
  if (observable.t < start_time || observable.t >= end_time) return;
  ContextBase & begin = *observable.context;
  begin.set(min);
  update(begin);
}

void Analysis::when_unsubscribed_from(Observable & observable) {
  finalize();
}
