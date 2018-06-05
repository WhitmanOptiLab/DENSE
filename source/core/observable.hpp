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

  public:

    Observable() = default;

    void addObserver(Observer *);

    // Called by Observer in update
    void abort() { abort_signaled = true; }

    virtual void run() = 0;

    //"abort_signaled" condition checked
    void notify(ContextBase& start);

    void finalize();

  protected:

    std::vector<std::reference_wrapper<Observer>> observers_;

    double t = 0.0;

    bool abort_signaled = false;

};

/**
Superclass for CSV Writer and Analysis
*/
class Observer {

  public:

    Observer(Observable * oAble, int mn, int mx, RATETYPE startT, RATETYPE endT);

    virtual ~Observer() = default;

    int getMin() const;

    bool accepts_updates_at_time(double t) const;

    void try_update(double t, ContextBase &);

    virtual void finalize() = 0;

    virtual void update(ContextBase& start) = 0;

  protected:

    Observable * subject;

    RATETYPE start_time, end_time;

    int min, max;

};

#endif
