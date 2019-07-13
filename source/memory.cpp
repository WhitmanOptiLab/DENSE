/*
Stochastically ranked evolutionary strategy sampler for zebrafish segmentation
Copyright (C) 2013 Ahmet Ay, Jack Holland, Adriana Sperlea, Sebastian Sangervasi

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

/*
memory.cpp contains functions related to memory management. All memory related functions should be placed in this file.
Many features and functions are enabled only when scons-compiling with 'memtrack=1', which defines the MEMTRACK macro used for memory tracking.
*/

#include "memory.hpp" // Function declarations

#include "macros.hpp"
#include "structs.hpp"

extern terminal* term; // Declared in init.cpp

// Variables the memory tracker uses to keep track of heap usage
#if defined(MEMTRACK)
	size_t heap_current = 0;
	size_t heap_total = 0;
#endif

/* mallocate allocates a block of memory with the given size
	parameters:
		size: the number of bytes to allocate
	returns: a pointer to the block of memory allocated
	notes:
		This function is a thin wrapper for malloc that exits if the memory cannot be allocated or a nonpositive size is given.
		If memory tracking is enabled, extra bytes are allocated with every request to store the size of the request. The memory tracker does not count these extra bytes when reporting heap usage.
		Memory allocated with mallocate should be freed with mfree, not free.
	todo:
*/
void* mallocate (size_t size) {
	if (size > 0) {
		void* block;
		#if defined(MEMTRACK)
			block = malloc(sizeof(size_t) + size);
		#else
			block = malloc(size);
		#endif
		if (block == NULL) {
			term->no_memory();
			exit(EXIT_MEMORY_ERROR);
		}
		#if defined(MEMTRACK)
			heap_current += size;
			heap_total += size;
			size_t* sizeblock = (size_t*)block;
			*sizeblock = size;
			return (void*)(sizeblock + 1);
		#else
			return block;
		#endif
	} else {
		cout << term->red << "The specified amount of memory to allocate (" << size << " B) must be a positive integer!" << term->reset << endl;
		exit(EXIT_MEMORY_ERROR);
	}
}

/* callocate allocates a block of memory with a size equal to the given (elements * size) parameters and zeros out the block
	parameters:
		elements: the number of elements to be allocated
		size: the size of each element to allocate
	returns: a pointer to the block of memory allocated
	notes:
		This function is a thin wrapper for calloc that uses mallocate instead of malloc.
		Memory allocated with callocate should be freed with mfree, not free.
		This function exists because libSRES used calloc.
	todo:
*/
void* callocate (size_t elements, size_t size) {
	void* mem = mallocate(elements * size);
	memset(mem, 0, elements * size);
	return mem;
}

/* reallocate allocates a block of memory with the given size or reuses the given block if it is large enough
	parameters:
		elements: the number of elements to be allocated
		size: the size of each element to allocate
	returns: a pointer to the block of memory allocated
	notes:
		This function is a thin wrapper for realloc that uses mallocate instead of malloc.
		Memory allocated with reallocate should be freed with mfree, not free.
		This function exists because libSRES used realloc.
	todo:
*/
void* reallocate (void* mem, size_t size) {
	#if defined(MEMTRACK)
		if (mem == NULL) {
			return mallocate(size);
		} else {
			size_t* sizeblock = (size_t*)mem - 1;
			if (*sizeblock < size) {
				void* newmem = mallocate(size);
				memcpy(newmem, mem, *sizeblock);
				mfree(sizeblock);
				return newmem;
			} else {
				return mem;
			}
		}
	#else
		return realloc(mem, size);
	#endif
}

/* mfree frees the given block of memory
	parameters:
		mem: a pointer to the block of memory to free
	returns: nothing
	notes:
		This function is a thin wrapper for free that ensures the extra bytes allocated when memory tracking is on are also freed.
		Always use this function to free memory allocated with mallocate since free will not work properly when memory tracking is on.
	todo:
*/
void mfree (void* mem) {
	#if defined(MEMTRACK)
		if (mem != NULL) {
			size_t* memblock = (size_t*)mem - 1;
			heap_current -= *memblock;
			free(memblock);
		}
	#else
		free(mem);
	#endif
}

/* new overloads the usual new with mallocate instead of malloc
	parameters:
		size: the number of bytes to allocate [ this parameter is not inputted directly due to the syntax of new; new int(x) translates conceptually to new(sizeof(int)) ]
	returns: a pointer to the block of memory allocated
	notes:
		This function forces all memory allocation through mallocate, which allows custom error reporting and memory tracking.
	todo:
*/
void* operator new (size_t size) {
	return mallocate(size);
}

/* new[] overloads the usual new[] with mallocate instead of malloc
	parameters:
		size: the number of bytes to allocate [ this parameter is not inputted directly due to the syntax of new[]; new int[x] translates conceptually to new(sizeof(int) * x) ]
	returns: a pointer to the block of memory allocated
	notes:
		This function forces all memory allocation through mallocate, which allows custom error reporting and memory tracking.
	todo:
*/
void* operator new[] (size_t size) {
	return mallocate(size);
}

/* delete overloads the usual delete with mfree instead of free
	parameters:
		mem: a pointer to the block of memory to free
	returns: nothing
	notes:
		This function forces all memory deallocation through mfree, which allows custom error reporting and memory tracking.
	todo:
*/
void operator delete (void* mem) {
	mfree(mem);
}

/* delete[] overloads the usual delete[] with mfree instead of free
	parameters:
		mem: a pointer to the block of memory to free
	returns: nothing
	notes:
		This function forces all memory deallocation through mfree, which allows custom error reporting and memory tracking.
	todo:
*/
void operator delete[] (void* mem) {
	mfree(mem);
}

#if defined(MEMTRACK)

/* print_mem_amount prints the given number of bytes in a human-friendly format
	parameters:
		mem: the number of bytes to print
	returns: nothing
	notes:
	todo:
*/
static void print_mem_amount (size_t mem) {
	static size_t kB = 1024;
	static size_t MB = SQUARE(1024);
	static size_t GB = CUBE(1024);
	double dmem = mem;
	
	if (mem > GB) {
		cout << dmem / GB << " GB";
	} else if (mem > MB) {
		cout << dmem / MB << " MB";
	} else if (mem > kB) {
		cout << dmem / kB << " kB";
	} else {
		cout << dmem << " B";
	}
	cout << endl;
}

/* print_heap_usage prints the current and total heap usage calculated with the memory tracker
	parameters:
	returns: nothing
	notes:
		Current heap usage indicates how much unfreed memory is on the heap.
		Total heap usage indicates how much memory has been allocated since the program's inception.
		Do not call this function after free_terminal or reset_cout since it uses terminal colors allocated by init_terminal and quiet mode does not work after reset_cout.
	todo:
*/
void print_heap_usage () {
	cout << term->blue << "Current heap usage:\t" << term->reset;
	print_mem_amount(heap_current);
	cout << term->blue << "Total heap usage:\t" << term->reset;
	print_mem_amount(heap_total);
}

#endif

