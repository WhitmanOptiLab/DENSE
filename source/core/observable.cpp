#include "observable.hpp"


void Observable::notify () {
  for (Observer & subscriber : subscribers()) {
    subscriber.when_updated_by(*this);
  }
}

void Observer::subscribe_to(Observable & observable) {
  subscriptions_.emplace_back(observable);
  observable.subscribers_.emplace_back(*this);
  when_subscribed_to(observable);
}

void Observer::unsubscribe_from_all() {
  for (Observable & subscription : subscriptions()) {
    unsubscribe_from(subscription);
  }
}

void Observer::unsubscribe_from(Observable & observable) {
  when_unsubscribed_from(observable);
}

std::vector<std::reference_wrapper<Observable>> const& Observer::subscriptions() {
  return subscriptions_;
}

void Observer::when_updated_by(Observable & observable) {
}
