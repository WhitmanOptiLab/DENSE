#include <vector>
#include <set>
#include "reaction.hpp"

using namespace std;


class Queue {

private:
	vector<RATETYPE> contents;
	int start,end,midpoint,size;
	
	void recalc_midpoint(){
		if (end>start){
			midpoint = ((end-start)/2)+start;
		}else if(start>end){
			int mid_temp = start+(((size+end)-start)/2);
			if (mid_temp>=size){
				midpoint = mid_temp-size;
			}else{
				midpoint = mid_temp;
			}
		}else{
			midpoint = -1;
		}
	}

public:
//	Queue();
	
	Queue(int length){
		contents.resize(length);
		size = length;
		start = 0;
		midpoint = -1;
		end = 0;
	}
/*
	void populate(RATETYPE transferArray[]){
		
		if (isEmpty()){
			for (int i=0; i<size/2; i++){
				contents[i] = transferArray[i];
			}
			start = 0;
			midpoint = size/4;
			end = (size/2)-1;
		}
	}

	bool isEmpty(){
		return (midpoint == -1);
	}
*/
	void enqueue(RATETYPE entry){
		if (end == (size-1)){
			end=0;
		}
		else{
			end++;
		}

		contents[end] = entry;

		recalc_midpoint();
	}

	RATETYPE dequeue(){
		RATETYPE popped = contents[start];
		if (start == (size-1)){
			start = 0;
		}
		else{
			start++;
		}		
		recalc_midpoint();
	
		return popped;
	}
	
	RATETYPE getVal(int index){
		return contents[index];
	}	

	RATETYPE getMidpoint(){
		return midpoint;
	}

	void print(){
		cout<<"["<<contents[0];
		for (int i=1; i<size; i++){
			cout<<","<<contents[i];
		}
		cout<<"]"<<endl;
	}
};
