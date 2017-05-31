#ifndef OBSERVER
#define OBSERVER

#include <iostream>
#include <vector>

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

	void notify();
};

/**
Superclass for DataLogger
*/
class Observer{

	protected:			
	Observable *subject;

	public:
	Observer(Observable *oAble);
	virtual void update() = 0;
};
#endif

