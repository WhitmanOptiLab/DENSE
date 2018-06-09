#ifndef CORE_OBSERVABLE_HPP_INCLUDED
#define CORE_OBSERVABLE_HPP_INCLUDED

#include <functional>
#include <vector>

#include "context.hpp"


class Observer;

/**
Superclass for Simulation
*/
class Observable {

  friend Observer;

  public:

    virtual void run() = 0;

    void notify(ContextBase& start);

    void finalize();

    double t = 0.0;

    ContextBase * context = nullptr;

  protected:

    std::vector<std::reference_wrapper<Observer>> const& subscribers() const {
      return subscribers_;
    }

  private:

    std::vector<std::reference_wrapper<Observer>> subscribers_;

};

/**
Superclass for CSV Writer and Analysis
*/
class Observer {

  friend Observable;

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
    virtual ~Observer () { unsubscribe_from_all(); };

    /// Subscribe an observer to an observable.
    /// \post `(*subscriptions().rbegin()).get() == observable`
    void subscribe_to (Observable & observable);

    /// Unsubscribe an observer from an observable.
    /// \complexity Constant if \c observable is the most recent subscription;
    ///             linear in the number of subscriptions otherwise.
    void unsubscribe_from (Observable & observable);

    /// Unsubscribe an observer from all of its subscriptions.
    /// \complexity Linear in the number of subscriptions.
    void unsubscribe_from_all ();

    /// View an observer's subscriptions as a ContiguousContainer.
    /// \complexity Constant.
    std::vector<std::reference_wrapper<Observable>> const& subscriptions ();

    bool operator==(Observer & other) const noexcept {
      return this == &other;
    }

    bool operator!=(Observer & other) const noexcept {
      return !operator==(other);
    }

    bool operator<(Observer & other) const noexcept {
      return this < &other;
    }

  protected:

    virtual void when_subscribed_to (Observable &) {};

    virtual void when_updated_by (Observable &);

    virtual void when_unsubscribed_from (Observable &) {};

  private:

    std::vector<std::reference_wrapper<Observable>> subscriptions_;

};

#endif
