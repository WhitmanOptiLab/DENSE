#ifndef CORE_OBSERVER_HPP
#define CORE_OBSERVER_HPP

#include <iostream>
#include <vector>
#include "context.hpp"


class Observer;

/**
Superclass for Simulation
*/
class Observable{
    
    protected:
	std::vector<Observer*> observerList;
    bool abort_signaled;
    double t;

	public:
    Observable() : t(0.0), abort_signaled(false) {}

	void addObserver(Observer *o){
		observerList.push_back(o);
	}

    //Called by Observer in update
    void abort(){abort_signaled = true;}
    
    virtual void run() = 0;

    //"abort_signaled" condition checked
	void notify(ContextBase& start);
	void finalize();
};

/**
Superclass for CSV Writer and Analysis
*/
class Observer{

	protected:			
	Observable *subject;
    int min, max;
    RATETYPE start_time, end_time;

	public:
	Observer(Observable *oAble, int mn, int mx, RATETYPE startT, RATETYPE endT);
	virtual ~Observer() {}
    int getMin();
    bool isInTimeBounds(double t){
        return t >= start_time && t < end_time;
    }

	virtual void finalize() = 0;
    virtual void update(ContextBase& start) = 0;
};
#endif
