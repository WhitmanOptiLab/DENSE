#include "set_cuda.hpp"
#include <iostream>
#include <limits>
#include <cmath>

namespace{
__global__ void executeAll(simulation_cuda* _sim_set) {
   unsigned int k = threadIdx.x;
   unsigned int set = blockIdx.x;
   _sim_set[set].execute_one(k);
}
}

void simulation_set_cuda::simulate_sets() {
    RATETYPE total_step = time_total/_sim_set[0]._step_size;
    //Set dimensions
    dim3 dimBlock(_sim_set[0]._cells_total,1,1); //each cell had own thread

    //dim3 dimGrid(1,1,1); //simulation done on single block
    dim3 dimGrid(_num_sets,1,1);

    //cudaDeviceSetLimit(cudaLimitStackSize, 65536);
    //Run kernel
    //cout.precision(dbl::max_digits10);
    for (int i=0;i<total_step;i++){
//        cout << i << ' ';
//        for (int j = 0; j < 2; j++) {
//           cout<< _sim_set[j]._baby_cl_cuda[ph11][i][0]<<", ";
//        }
//        cout<< endl;

        executeAll<<<dimGrid, dimBlock>>>(_sim_set);
        //cudaDeviceSynchronize(); //Required to be able to access managed
                                 // GPU data
    }

    cudaDeviceSynchronize();
    //convert back to CPU
    if (cudaPeekAtLastError() != cudaSuccess) {
        std::cout << "Kernel launch error: " << cudaPeekAtLastError() << "\n";
    }

    cudaDeviceSynchronize();
}
