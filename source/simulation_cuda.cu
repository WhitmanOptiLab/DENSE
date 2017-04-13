#include <cmath>
#include "simulation_cuda.hpp"
#include "cell_param.hpp"
#include "context.hpp"
#include "context_impl.hpp"
#include <limits>
#include <iostream>

#define CUDA_ERRCHK(call) (call)

typedef std::numeric_limits<double> dbl;
using namespace std;

void simulation_cuda::initialize(){
    calc_max_delays(); 
    _delays.update_rates(_parameter_set._delay_sets);
    _rates.update_rates(_parameter_set._rates_base);
    _critValues.update_rates(_parameter_set._critical_values);
    _cl.initialize(4,300,200);
    _baby_cl_cuda.initialize();
}

namespace {
    __global__ void cudasim_execute(simulation_cuda _sim_cu){

        unsigned int k = threadIdx.x;

        // Iterate through each extant cell or context
        if (_sim_cu._width_current == _sim_cu._width_total || k % _sim_cu._width_total <= 10) { // Compute only existing (i.e. already grown)cells
            // Calculate the cell indices at the start of each mRNA and protein's dela
            Context c(_sim_cu, k);

            // Perform biological calculations
            c.updateCon(c.calculateRatesOfChange());
        }

        if (k==0){
            _sim_cu._j++;
        }
        if (k < NUM_SPECIES) {
            _sim_cu._baby_j[k]++;
        }
    }
} // end namespace


void simulation_cuda::simulate_cuda(RATETYPE sim_time){
    RATETYPE total_step = sim_time/_step_size;

    //Set dimensions
    dim3 dimBlock(_cells_total,1,1); //each cell had own thread

    //dim3 dimGrid(1,1,1); //simulation done on single block
    dim3 dimGrid(total_step,1,1);

    //cudaDeviceSetLimit(cudaLimitStackSize, 65536);
    //Run kernel
    for (int i=0;i<total_step;i++){
        cudasim_execute<<<dimGrid, dimBlock>>>(*this);
    }


    CUDA_ERRCHK(cudaDeviceSynchronize());
    //convert back to CPU
    if (cudaPeekAtLastError() != cudaSuccess) {
        cout << "Kernel launch error: " << cudaPeekAtLastError() << "\n";
    }

    cudaDeviceSynchronize();
}

