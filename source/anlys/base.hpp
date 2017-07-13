#ifndef ANLYS_BASE_HPP
#define ANLYS_BASE_HPP

#include "core/observable.hpp"

/*
* Superclass for Analysis Objects
* - observes passed "Observable"
* - does not implement any analysis
*/
class Analysis : public Observer {
protected:
	int time;
public:
	Analysis(Observable *dLog, int mn, int mx, RATETYPE startT, RATETYPE endT) 
        : Observer(dLog,mn,mx,startT,endT), time(0) {}

};

#endif
