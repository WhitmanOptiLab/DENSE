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
    dim3 dimGrid(6,1,1); //simulation done on single block


    cudaDeviceSetLimit(cudaLimitStackSize, 65536);
    //Run kernel
    //execute<<<dimGrid,dimBlock>>>(st,critical,cx,cHOR,cells);
    execute<<<dimGrid, dimBlock>>>();

    CUDA_ERRCHK(cudaDeviceSynchronize());
    //convert back to CPU
    if (cudaPeekAtLastError() != cudaSuccess) {
        cout << "Kernel launch error: " << cudaPeekAtLastError() << "\n";
    }

    cudaDeviceSynchronize();
}

__global__ void execute(){

    unsigned int k = threadIdx.x;
    unsigned int i = blockIdx.x;
    baby_cls[i].cons[BIRTH][baby_j][k] = baby_cls[i].cons[BIRTH][time_prev][k];
    baby_cls[i].cons[PARENT][baby_j][k] = baby_cls[i].cons[PARENT][time_prev][k];


    if (sd.width_current == sd.width_total || k % sd.width_total <= sd.active_start) { // Compute only existing (i.e. already grown) cells
    // Calculate the cell indices at the start of each mRNA and protein's delay
    int old_cells_mrna[NUM_INDICES];
    int old_cells_protein[NUM_INDICES];
    calculate_delay_indices(sd, baby_cls[i], baby_j, j, k, rates_active_arr[i], old_cells_mrna, old_cells_protein);

    // Perform biological calculations
    st_context stc(time_prev, baby_j, k);
    protein_synthesis(sd, rates_active_arr[i], baby_cls[i], stc, old_cells_protein);
    dimer_proteins(sd,rates_active_arr[i], baby_cls[i], stc);
    mRNA_synthesis(sd, rates_active_arr[i], baby_cls[i], stc, old_cells_mrna, p[i], past_induction[i], past_recovery[i]);
    }

}
