#include "observable.hpp"

void Observable :: notify(ContextBase& start){
	for (int i=0; i<observerList.size(); i++){
		start.set(observerList[i]->getMin());
		if (observerList[i]->isInTimeBounds(t)){
            observerList[i]->update(start);
        }
	}
}

void Observable :: finalize(){
	for (int i = 0; i<observerList.size(); i++){	
        observerList[i]->finalize();
	}
}		

Observer :: Observer(Observable *oAble, int mn, int mx, 
                       RATETYPE startT, RATETYPE endT) :
    min(mn), max(mx), start_time(startT), end_time(endT), subject(oAble) {
	
    subject->addObserver(this);
}

int Observer :: getMin(){
    return min;
}
