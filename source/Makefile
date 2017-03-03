CC=g++
NVCC=nvcc
CFLAGS=-O3 -std=c++11
LFLAGS=${CFLAGS}
TESTS=test_simulation

test: test_simulation
	./test_sim_compile 
	
test_sim_compile: test_sim_compile.o simulation.o baby_cl.o rates.o model.o reaction.o
	${CC} ${CFLAGS} $^ -o $@ 

%.o: %.cpp $(wildcard *.h)
	${CC} ${CFLAGS} -c $< -o $@

clean:
	rm -f ${TESTS} ./*.o

