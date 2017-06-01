MODEL=her_model_2014
ifndef MODEL
$(error Please define the variable MODEL to be one of ($(shell ls ../models)))
endif

CC=g++
NVCC=nvcc
CFLAGS=-g -std=c++11 -I../models/${MODEL} -I.
LFLAGS=${CFLAGS}
CUFLAGS=-dc ${CFLAGS} -arch=sm_30
CULFLAGS=${CFLAGS} -arch=sm_30 -lcuda -lcudart_static
CPU_TESTS=test_simulation test_simulation_set 
CUDA_TESTS=test_simulation_cuda test_simulation_set_cuda
HEADERS=$(wildcard *.hpp) $(wildcard ../models/${MODEL}/*.hpp)


all: ${CPU_TESTS}

gen_csv: gen_csv.o 
	${CC} ${LFLAGS} $^ -o $@ 

cpu_test: ${CPU_TESTS}
	./test_simulation
	./test_simulation_set

cuda_test: ${CUDA_TESTS}
	./test_simulation_cuda
	./test_simulation_set_cuda

test: cpu_test cuda_test
	
test_sim_compile: test_sim_compile.o simulation.o baby_cl.o cell_param.o model.o reaction.o observable.o color.o   
	${CC} ${LFLAGS} $^ -o $@ 

test_simulation: test_simulation.o simulation.o reaction.o cell_param.o baby_cl.o model.o observable.o param_set.o arg_parse.o color.o
	${PARAM_SETS} ${CC} ${LFLAGS} $^ -o $@ 

test_simulation_cuda: test_simulation_cuda.o simulation.o simulation_cuda.o reaction_cu.o cell_param.o baby_cl_cuda.o model.o baby_cl.o observable.o color.o 
	${NVCC} ${CULFLAGS} $^ -o $@ 

test_simulation_set: test_simulation_set.o simulation.o reaction.o cell_param.o baby_cl.o model.o observable.o param_set.o arg_parse.o color.o
	${CC} ${LFLAGS} $^ -o $@ 

test_simulation_set_cuda: test_simulation_set_cuda.o simulation_set_cuda.o simulation.o simulation_cuda.o reaction_cu.o cell_param.o baby_cl.o baby_cl_cuda.o model.o observable.o color.o 

	${NVCC} ${CULFLAGS} $^ -o $@ 

reaction_cu.o: ../models/${MODEL}/reaction.cu $(HEADERS)
	${NVCC} ${CUFLAGS} -c $< -o $@

reaction.o: ../models/${MODEL}/reaction.cpp $(HEADERS)
	${CC} ${CFLAGS} -c $< -o $@

%.o: %.cpp $(wildcard *.h) $(HEADERS)
	${CC} ${CFLAGS} -c $< -o $@

%.o: %.cu $(wildcard *.h) $(HEADERS)
	${NVCC} ${CUFLAGS} -c $< -o $@

clean:
	rm -f ${TESTS} ./*.o

