#include "observable.hpp"


void Observable::notify(ContextBase& start) {
  for (Observer & subscriber : subscribers()) {
    subscriber.try_update(t, start);
  }
}

void Observable::finalize() {
  for (Observer & subscriber : subscribers()) {
    subscriber.finalize();
  }
}

Observer::Observer(
  Observable *observable,
  int min, int max,
  RATETYPE start_time,
  RATETYPE end_time
) :
  min{min},
  max{max},
  start_time{start_time},
  end_time{end_time},
  subject(*observable)
{
  subscribe_to(subject);
}

void Observer::subscribe_to(Observable & observable) {
  subject = observable;
  observable.subscribers_.emplace_back(*this);
}

void Observer::try_update(double t, ContextBase & begin) {
  if (t < start_time || t >= end_time) return;
  begin.set(min);
  update(begin);
}
