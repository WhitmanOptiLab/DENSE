#include "observable.hpp"

void Observable :: notify(ContextBase& start){
	for (int i=0; i<observerList.size(); i++){
		observerList[i]->update(start);
		start.reset();
	}
}

void Observable :: notify(ContextBase& start, bool isFinal){
	for (int i = 0; i<observerList.size(); i++){	
		if (isFinal){observerList[i]->finalize(start);}
		else{observerList[i]->update(start);}
		start.reset();
	}
}		

Observer :: Observer(Observable *oAble){
	subject = oAble;
	subject->addObserver(this);
}

void Observer :: finalize(ContextBase& start){}
