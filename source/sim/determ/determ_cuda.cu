#include <cmath>
#include "determ_cuda.hpp"
#include "sim/cell_param.hpp"
#include "model_impl.hpp"
#include "determ_cuda_context.hpp"
#include <limits>
#include <iostream>

void simulation_cuda::initialize(){
    Deterministic_Simulation::initialize();
    _baby_cl_cuda.initialize();
}

namespace {
    __global__ void cudasim_execute(simulation_cuda& _sim_cu){

        unsigned int k = threadIdx.x;
        _sim_cu.execute_one(k);
    }
} // end namespace


void simulation_cuda::simulate_cuda(){
    RATETYPE analysis_chunks = time_total/analysis_gran;
    RATETYPE total_step = analysis_gran/_step_size;
    //Set dimensions
    dim3 dimBlock(_cells_total,1,1); //each cell had own thread

    //dim3 dimGrid(1,1,1); //simulation done on single block
    dim3 dimGrid(1,1,1);

    //cudaDeviceSetLimit(cudaLimitStackSize, 65536);
    //Run kernel
    std::cout.precision(std::numeric_limits<Real>::max_digits10);
    for (int c=0; c<analysis_chunks; c++){
    	for (int i=0;i<total_step;i++){
        	std::cout << _j << " " << _baby_cl_cuda[ph11][0][0] << '\n';
        	cudasim_execute<<<dimGrid, dimBlock>>>(*this);

        	cudaDeviceSynchronize(); //Required to be able to access managed
                	                 // GPU data
    	}
    }

    check(cudaDeviceSynchronize());
    //convert back to CPU
    if (cudaPeekAtLastError() != cudaSuccess) {
        std::cout << "Kernel launch error: " << cudaPeekAtLastError() << '\n';
    }

    cudaDeviceSynchronize();
}
