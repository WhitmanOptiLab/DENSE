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

    Observable() = default;

    virtual void run() = 0;

    //"abort_signaled" condition checked
    void notify(ContextBase& start);

    void finalize();

  protected:

    std::vector<std::reference_wrapper<Observer>> const& observers() const {
      return observers_;
    }

    double t = 0.0;

    bool abort_signaled = false;

  private:

    // Called by Observer in update
    void abort() { abort_signaled = true; }

    std::vector<std::reference_wrapper<Observer>> observers_;

};

/**
Superclass for CSV Writer and Analysis
*/
class Observer {

  public:

    Observer(Observable * observable, int min, int max, RATETYPE start_time, RATETYPE end_time);

    virtual ~Observer() = default;

    void try_update(double t, ContextBase &);

    void subscribe_to(Observable &);

    virtual void finalize() = 0;

    virtual void update(ContextBase& start) = 0;

  protected:

    Observable & subject;

    RATETYPE start_time, end_time;

    int min, max;

};

#endif
