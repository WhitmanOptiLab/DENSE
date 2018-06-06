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

    virtual ~Observer() = default;

    void subscribe_to(Observable &);

  protected:

    virtual void try_update(Observable &);

    virtual void finalize() = 0;

  private:

    std::vector<std::reference_wrapper<Observable>> subscriptions_;

};

/*
  Observer that restricts updates to a specific range of cells and times
  Used as a stepping-stone to refactoring the Observer/Observable interface
*/
class PickyObserver : public Observer {

  public:

    PickyObserver(Observable & observable, int min, int max, RATETYPE start_time, RATETYPE end_time);

    void try_update(Observable &) override;

    virtual void update(ContextBase& start) = 0;

  protected:

    Real start_time, end_time;

    int min, max;

};

#endif
