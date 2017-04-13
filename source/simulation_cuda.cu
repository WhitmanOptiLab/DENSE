#include <cmath>
#include "simulation_cuda.hpp"
#include "cell_param.hpp"
#include "context.hpp"
#include <limits>
#include <iostream>

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


__global__ void simulation_cuda::simulate_cuda(RATETYPE sim_time){
    RATETYPE total_step = sim_time/_step_size;
    cudaEvent_t start,stop;

    //Set dimensions
    dim3 dimBlock(_cells_total,1,1); //each cell had own thread

    //dim3 dimGrid(1,1,1); //simulation done on single block
    dim3 dimGrid(total_step,1,1);

    //cudaDeviceSetLimit(cudaLimitStackSize, 65536);
    //Run kernel
    for (int i=0;i<total_step;i++){
        execute<<<dimGrid, dimBlock>>>();
    }


    CUDA_ERRCHK(cudaDeviceSynchronize());
    //convert back to CPU
    if (cudaPeekAtLastError() != cudaSuccess) {
        cout << "Kernel launch error: " << cudaPeekAtLastError() << "\n";
    }

    cudaDeviceSynchronize();
}

__global__ void execute(){

    unsigned int k = threadIdx.x;

    int steps_elapsed = steps_split;
    //cout.precision(dbl::max_digits10);
    //cout<< _j<< " "<<_baby_cl[ph11][_j][1]<<endl;
    // Iterate through each extant cell or context

    if (_width_current == _width_total || k % _width_total <= 10) { // Compute only existing (i.e. already grown)cells
        // Calculate the cell indices at the start of each mRNA and protein's dela
        Context c(*this, k);
#if 0
        int old_cells_mrna[NUM_SPECIES];
        int old_cells_protein[NUM_SPECIES]; // birth and parents info are kept elsewhere now

        // Perform biological calculations
        c.updateCon(c.calculateRatesOfChange());
    }


    _j++;
    for (int i =0; i< NUM_SPECIES ; i++){
        _baby_j[i]++;
    }

}
