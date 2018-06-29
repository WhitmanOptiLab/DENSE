#ifndef CORE_QUEUE_HPP
#define CORE_QUEUE_HPP

#include "utility/numerics.hpp"

#include <iostream>
#include <vector>

template <typename T>
class Queue {

  using value_type = T;
  using index_type = std::ptrdiff_t;
  using size_type = index_type;
  using reference = T&;
  using const_reference = T const&;
  using iterator = T*;
  using const_iterator = T const*;

private:
	std::vector<T> contents;
	index_type start,end,current;
  index_type size;
  index_type capacity;

public:
//	Queue();

	Queue(index_type length) {
		contents.resize(length + 1);
		size = capacity = length + 1;
		start = 0;
		current = -1;
		end = -1;
	}
/*
	void populate(Real transferArray[]){

		if (isEmpty()){
			for (dense::Natural i = 0; i < size/2; ++i){
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

	size_type getSize() { return (size + end - start) % size + 1; }

	void enqueue(value_type entry){
		if (end == (size-1)){
			end = 0;
		}
		else {
			++end;
		}
		contents[end] = entry;

		if (2*getSize() >= size) {
			++current;
			if (current == size) {
				current = 0;
			}
		}
	}

	value_type dequeue(){
		auto popped = contents[start];
		if (start == (size-1)){
			start = 0;
		}
		else {
			++start;
		}
		if (2*getSize() <= size) {
			++current;
			if (current == size) {
				current = 0;
			}
		}
		return popped;
	}

	value_type getVal(dense::Natural index){
		return contents[index];
	}

	index_type getCurrent() {
		return current;
	}

	friend std::ostream & operator << (std::ostream & out, Queue<T> & queue) {
		out << '[' << queue.contents[0];
		for (Queue<T>::size_type i = 1; i < queue.size; ++i) {
			out << ',' << queue.contents[i];
		}
		return out << "]\n";
	}
};

#endif
