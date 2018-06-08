#include "base.hpp"

Analysis::Analysis (
  Observable * log,
  specie_vec const& species_vector,
  csvw * csv_out,
  int min, int max,
  Real start_time, Real end_time
) :
  PickyObserver(*log, min, max, start_time, end_time),
  time{0},
  ucSpecieOption(species_vector),
  csv_out(csv_out) {
};

PickyObserver::PickyObserver(
  Observable & observable,
  int min, int max,
  RATETYPE start_time,
  RATETYPE end_time
) :
  min{min},
  max{max},
  start_time{start_time},
  end_time{end_time}
{
  subscribe_to(observable);
}

void Analysis::when_updated_by(Observable & observable) {
  if (observable.t < start_time || observable.t >= end_time) return;
  ContextBase & begin = *observable.context;
  begin.set(min);
  update(begin);
}

void Analysis::when_unsubscribed_from(Observable & observable) {
  finalize();
}
