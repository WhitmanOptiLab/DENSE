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
	Analysis(Observable *dLog) : Observer(dLog) {
		time = 0;
	}

};

#endif
