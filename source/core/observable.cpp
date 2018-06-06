#include "observable.hpp"


void Observable::notify(ContextBase& start) {
  context = &start;
  for (Observer & subscriber : subscribers()) {
    subscriber.try_update(*this);
  }
  context = nullptr;
}

void Observable::finalize() {
  for (Observer & subscriber : subscribers()) {
    subscriber.finalize();
  }
}

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

void Observer::subscribe_to(Observable & observable) {
  subscriptions_.emplace_back(observable);
  observable.subscribers_.emplace_back(*this);
}

void Observer::try_update(Observable & observable) {
}

void PickyObserver::try_update(Observable & observable) {
  if (observable.t < start_time || observable.t >= end_time) return;
  ContextBase & begin = *observable.context;
  begin.set(min);
  update(begin);
}
