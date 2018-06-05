#include "observable.hpp"


void Observable::notify(ContextBase& start) {
  for (Observer & observer : observers_) {
    observer.try_update(t, start);
  }
}

void Observable::finalize() {
  for (Observer & observer : observers_) {
    observer.finalize();
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
  observable.observers_.emplace_back(*this);
}

void Observer::try_update(double t, ContextBase & begin) {
  if (t < start_time || t >= end_time) return;
  begin.set(min);
  update(begin);
}
