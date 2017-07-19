#ifndef ANLYS_BASE_HPP
#define ANLYS_BASE_HPP

#include "core/observable.hpp"
#include "core/specie.hpp"
#include "io/csvw.hpp"

/*
* Superclass for Analysis Objects
* - observes passed "Observable"
* - does not implement any analysis
*/
class Analysis : public Observer {
protected:
	int time;
    const specie_vec ucSpecieOption;
    csvw* unFileOut;

public:
	Analysis(Observable *dLog, const specie_vec& pcfSpecieOption, csvw* pnFileOut,
            int mn, int mx, RATETYPE startT, RATETYPE endT) 
        : Observer(dLog,mn,mx,startT,endT), time(0), ucSpecieOption(pcfSpecieOption),
   unFileOut(pnFileOut) {}

};

#endif
