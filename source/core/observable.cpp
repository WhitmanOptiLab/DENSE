#include "observable.hpp"


void Observable::notify(ContextBase& start) {
  context = &start;
  for (Observer & subscriber : subscribers()) {
    subscriber.when_updated_by(*this);
  }
  context = nullptr;
}

void Observable::finalize() {
  for (Observer & subscriber : subscribers()) {
    subscriber.unsubscribe_from(*this);
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
  when_subscribed_to(observable);
}

void Observer::unsubscribe_from(Observable & observable) {
  when_unsubscribed_from(observable);
}

std::vector<std::reference_wrapper<Observable>> const& Observer::subscriptions() {
  return subscriptions_;
}

void Observer::when_updated_by(Observable & observable) {
}

void PickyObserver::when_updated_by(Observable & observable) {
  if (observable.t < start_time || observable.t >= end_time) return;
  ContextBase & begin = *observable.context;
  begin.set(min);
  update(begin);
}

void PickyObserver::when_unsubscribed_from(Observable & observable) {
  finalize();
}
