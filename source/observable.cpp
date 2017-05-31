#include "observable.hpp"

void Observable :: notify(){
	for (int i=0; i<observerList.size(); i++){
		observerList[i]->update();
	}
}

Observer :: Observer(Observable *oAble){
	subject = oAble;
	subject->addObserver(this);
}

