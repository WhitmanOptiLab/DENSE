#ifndef CORE_OBSERVABLE_HPP_INCLUDED
#define CORE_OBSERVABLE_HPP_INCLUDED

#include <functional>
#include <vector>

template <typename T>
class Observer;

/**
Superclass for Simulation
*/
template <typename T>
class Observable {

  friend Observer<T>;

  public:

    Observable() noexcept = default;

    Observable (Observable const&) = delete;

    Observable& operator= (Observable const&) = delete;

    Observable (Observable&& other) : subscribers_{std::move(other.subscribers_)} {

    }

    Observable& operator= (Observable&& other) {
      unsubscribe_all();
      using std::swap;
      swap(subscribers_, other.subscribers_);
      for (Observer<T>& subscriber : subscribers_) {
        for (auto& observable : subscriber.subscriptions_) {
          if (observable == other) observable = *this;
        }
      }
    }

    void notify ();

    void unsubscribe_all() noexcept {
      for (Observer<T>& subscriber : subscribers_) {
        subscriber.unsubscribe_from(*this);
      }
    }

    virtual ~Observable() noexcept = default;

  protected:

    std::vector<std::reference_wrapper<Observer<T>>> const& subscribers() const {
      return subscribers_;
    }

  private:

    std::vector<std::reference_wrapper<Observer<T>>> subscribers_;

};

/**
Superclass for CSV Writer and Analysis
*/
template <typename Observable_T>
class Observer {

  friend Observable<Observable_T>;

  public:

    /// Construct an observer with no subscriptions.
    /// \complexity Constant.
    Observer () = default;

    /// Forbid copy-constructing an observer.
    /// \warning Deleted.
    Observer (Observer const&) = delete;

    /// Forbid copy-assigning to an observer.
    /// \warning Deleted.
    Observer & operator= (Observer const&) = delete;

    /// Destruct an observer, unsubscribing it from all of its subscriptions.
    /// \complexity Linear in the number of subscriptions.
    virtual ~Observer () noexcept { unsubscribe_from_all(); };

    /// Subscribe an observer to an observable.
    /// \post `(*subscriptions().rbegin()).get() == observable`
    void subscribe_to (Observable_T & observable);

    /// Unsubscribe an observer from an observable.
    /// \complexity Constant if \c observable is the most recent subscription;
    ///             linear in the number of subscriptions otherwise.
    void unsubscribe_from (Observable_T & observable) noexcept;

    /// Unsubscribe an observer from all of its subscriptions.
    /// \complexity Linear in the number of subscriptions.
    void unsubscribe_from_all () noexcept;

    /// View an observer's subscriptions as a ContiguousContainer.
    /// \complexity Constant.
    std::vector<std::reference_wrapper<Observable_T>> const& subscriptions ();

    bool operator==(Observer const& other) const noexcept {
      return this == &other;
    }

    bool operator!=(Observer const& other) const noexcept {
      return !operator==(other);
    }

  protected:

    virtual void when_subscribed_to (Observable_T &) {};

    virtual void when_updated_by (Observable_T &) {};

    virtual void when_unsubscribed_from (Observable_T &) {};

  private:

    std::vector<std::reference_wrapper<Observable_T>> subscriptions_;

};


template <typename T>
void Observable<T>::notify () {
  for (Observer<T> & subscriber : subscribers()) {
    subscriber.when_updated_by(*static_cast<T*>(this));
  }
}

template <typename Observable_T>
void Observer<Observable_T>::subscribe_to(Observable_T & observable) {
  subscriptions_.emplace_back(observable);
  observable.subscribers_.emplace_back(*this);
  when_subscribed_to(observable);
}

template <typename Observable_T>
void Observer<Observable_T>::unsubscribe_from_all() noexcept {
  for (Observable_T & observable : subscriptions()) {
    when_unsubscribed_from(observable);
    for (auto& observer : observable.subscribers_) {
      if (observer.get() == *this) {
        observer = observable.subscribers_.back();
        observable.subscribers_.pop_back();
        break;
      }
    }
  }
  subscriptions_.clear();
}

template <typename Observable_T>
void Observer<Observable_T>::unsubscribe_from(Observable_T & observable) noexcept {
  when_unsubscribed_from(observable);
  for (auto& observer : observable.subscribers_) {
    if (observer.get() == *this) {
      observer = observable.subscribers_.back();
      observable.subscribers_.pop_back();
      break;
    }
  }
  for (auto& subscription : subscriptions_) {
    if (&observable == &subscription.get()) {
      subscription = subscriptions_.back();
      subscriptions_.pop_back();
      break;
    }
  }
}

template <typename Observable_T>
std::vector<std::reference_wrapper<Observable_T>> const& Observer<Observable_T>::subscriptions() {
  return subscriptions_;
}

#endif
