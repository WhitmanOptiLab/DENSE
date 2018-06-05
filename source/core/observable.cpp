#include "observable.hpp"

void Observable::addObserver(Observer * observer) {
  if (observer == nullptr) return;
  observers_.emplace_back(*observer);
}

void Observable::notify(ContextBase& start) {
  for (Observer & observer : observers_) {
    start.set(observer.getMin());
    if (observer.isInTimeBounds(t)) {
      observer.update(start);
    }
  }
}

void Observable::finalize() {
  for (Observer & observer : observers_) {
    observer.finalize();
  }
}

Observer :: Observer(Observable *oAble, int mn, int mx,
                       RATETYPE startT, RATETYPE endT) :
    min(mn), max(mx), start_time(startT), end_time(endT), subject(oAble) {

    subject->addObserver(this);
}

int Observer::getMin() const {
  return min;
}
