MODEL=her_model_2014
ifndef MODEL
$(error Please define the variable MODEL to be one of ($(shell ls ../models)))
endif

CC=g++
NVCC=/usr/local/cuda/bin/nvcc
CFLAGS=-g -std=c++11 -I../models/${MODEL} -I.
LFLAGS=${CFLAGS}
CUFLAGS=${CFLAGS} -arch=sm_30
CULFLAGS=${CFLAGS} -arch=sm_30 -lcuda -lcudart_static
TESTS=test_sim_compile test_simulation test_simulation_cuda test_simulation_set
HEADERS=$(wildcard *.hpp) $(wildcard ../models/${MODEL}/*.hpp)

all: ${TESTS}

test: ${TESTS}
	./test_sim_compile 
	./test_simulation
	./test_simulation_cuda
	./test_simulation_set
	
test_sim_compile: test_sim_compile.o simulation.o baby_cl.o cell_param.o model.o reaction.o 
	${CC} ${LFLAGS} $^ -o $@ 

test_simulation: test_simulation.o simulation.o reaction.o cell_param.o baby_cl.o model.o
	${CC} ${LFLAGS} $^ -o $@ 

test_simulation_cuda: test_simulation_cuda.o simulation.o simulation_cuda.o reaction.o cell_param.o baby_cl_cuda.o model.o baby_cl.o
	${NVCC} ${CULFLAGS} $^ -o $@ 

test_simulation_set: test_simulation_set.o simulation.o reaction.o cell_param.o baby_cl.o model.o
	${CC} ${LFLAGS} $^ -o $@ 

reaction.o: ../models/${MODEL}/reaction.cpp $(HEADERS)
	${CC} ${CFLAGS} -c $< -o $@

%.o: %.cpp $(wildcard *.h) $(HEADERS)
	${CC} ${CFLAGS} -c $< -o $@

%.o: %.cu $(wildcard *.h) $(HEADERS)
	${NVCC} ${CUFLAGS} -c $< -o $@

clean:
	rm -f ${TESTS} ./*.o

