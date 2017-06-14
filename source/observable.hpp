#ifndef OBSERVER
#define OBSERVER

#include <iostream>
#include <vector>

#include "context.hpp"

using namespace std;
class Observer;

/**
Superclass for Simulation
*/
class Observable{
	vector<Observer*> observerList;

	public:
	void addObserver(Observer *o){
		observerList.push_back(o);
	}

	void notify(ContextBase& start);
	void notify(ContextBase& start, bool isFinal);
};

/**
Superclass for DataLogger
*/
class Observer{

	protected:			
	Observable *subject;

	public:
	Observer(Observable *oAble);
	
	virtual void finalize(ContextBase& start) = 0;
	virtual void update(ContextBase& start) = 0;
};
#endif

