CC=g++
NVCC=nvcc
CFLAGS=-std=c++11
LFLAGS=${CFLAGS}
TESTS=test_sim_compile test_simulation

test: ${TESTS}
	./test_sim_compile 
	./test_simulation
	
test_sim_compile: test_sim_compile.o simulation.o baby_cl.o cell_param.o model.o reaction.o context.o
	${CC} ${CFLAGS} $^ -o $@ 

test_simulation: test_simulation.o simulation.o reaction.o cell_param.o baby_cl.o context.o
	${CC} ${CFLAGS} $^ -o $@ 

%.o: %.cpp $(wildcard *.h)
	${CC} ${CFLAGS} -c $< -o $@

clean:
	rm -f ${TESTS} ./*.o

