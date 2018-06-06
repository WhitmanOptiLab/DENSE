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

PickyObserver::PickyObserver(
  Observable *observable,
  int min, int max,
  RATETYPE start_time,
  RATETYPE end_time
) :
  min{min},
  max{max},
  start_time{start_time},
  end_time{end_time}
{
  subscribe_to(*observable);
}

void Observer::subscribe_to(Observable & observable) {
  subscriptions_.emplace_back(observable);
  observable.subscribers_.emplace_back(*this);
}

void Observer::try_update(double t, ContextBase & begin) {
  update(begin);
}

void PickyObserver::try_update(double t, ContextBase & begin) {
  if (t < start_time || t >= end_time) return;
  begin.set(min);
  Observer::try_update(t, begin);
}
