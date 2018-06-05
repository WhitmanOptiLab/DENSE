#include "observable.hpp"

void Observable::addObserver(Observer * observer) {
  observers_.push_back(observer);
}

void Observable::notify(ContextBase& start) {
  for (auto & observer : observers_) {
    start.set(observer->getMin());
    if (observer->isInTimeBounds(t)) {
      observer->update(start);
    }
  }
}

void Observable::finalize() {
  for (auto & observer : observers_) {
    observer->finalize();
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
