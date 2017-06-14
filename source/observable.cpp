#include "observable.hpp"

void Observable :: notify(ContextBase& start){
	for (int i=0; i<observerList.size(); i++){
		observerList[i]->update(start);
	}
}

Observer :: Observer(Observable *oAble){
	subject = oAble;
	subject->addObserver(this);
}
