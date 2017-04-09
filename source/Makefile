MODEL=her_model_2014
ifndef MODEL
$(error Please define the variable MODEL to be one of ($(shell ls ../models)))
endif

CC=clang++
NVCC=nvcc
CFLAGS=-O3 -std=c++11 -I../models/${MODEL} -I.
LFLAGS=${CFLAGS}
TESTS=test_sim_compile test_simulation test_simulation_set
HEADERS=$(wildcard *.hpp) $(wildcard ../models/${MODEL}/*.hpp)

all: ${TESTS}

test: ${TESTS}
	./test_sim_compile 
	./test_simulation
	./test_simulation_set
	
test_sim_compile: test_sim_compile.o simulation.o baby_cl.o cell_param.o model.o reaction.o context.o
	${CC} ${CFLAGS} $^ -o $@ 

test_simulation: test_simulation.o simulation.o reaction.o cell_param.o baby_cl.o context.o model.o
	${CC} ${CFLAGS} $^ -o $@ 

test_simulation_set: test_simulation_set.o simulation.o reaction.o cell_param.o baby_cl.o context.o model.o
	${CC} ${CFLAGS} $^ -o $@ 

reaction.o: ../models/${MODEL}/reaction.cpp $(HEADERS)
	${CC} ${CFLAGS} -c $< -o $@

%.o: %.cpp $(wildcard *.h) $(HEADERS)
	${CC} ${CFLAGS} -c $< -o $@

clean:
	rm -f ${TESTS} ./*.o

