#include <cmath>
#include "simulation_cuda.hpp"
#include "cell_param.hpp"
#include "model_impl.hpp"
#include "context_determ.hpp"
#include "cuda_context_impl.hpp"
#include <limits>
#include <iostream>

typedef std::numeric_limits<double> dbl;
using namespace std;

void simulation_cuda::initialize(){
    simulation_determ::initialize();
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
    cout.precision(dbl::max_digits10);
    for (int c=0; c<analysis_chunks; c++){
    	for (int i=0;i<total_step;i++){
        	cout<< _j<< " "<<_baby_cl_cuda[ph11][0][0]<<endl;
        	cudasim_execute<<<dimGrid, dimBlock>>>(*this);

        	cudaDeviceSynchronize(); //Required to be able to access managed 
                	                 // GPU data
    	}
    }

    check(cudaDeviceSynchronize());
    //convert back to CPU
    if (cudaPeekAtLastError() != cudaSuccess) {
        cout << "Kernel launch error: " << cudaPeekAtLastError() << "\n";
    }

    cudaDeviceSynchronize();
}

void simulation_cuda::calc_max_delays() {
  RATETYPE temp_delays[NUM_SPECIES];
  for (int s = 0; s < NUM_SPECIES; s++) {
    max_delays[s] = 0;
    temp_delays[s] = 0.0;
  }
  //for each reaction
  //  for each input
  //    accumulate delay into specie
  //  for each factor
  //    accumulate delay into specie
  //RATETYPE max_gradient_##name = 0; \
  //for (int k = 0; k < _width_total; k++) { \
  //  max_gradient_##name = std::max<int>(_model.factors_gradient[ name ][k], max_gradient_##name); \
  //} 
#define REACTION(name) 
#define DELAY_REACTION(name) \
  for (int in = 0; in < _model.reaction_##name.getNumInputs(); in++) { \
    specie_id sp = _model.reaction_##name.getInputs()[in]; \
    temp_delays[sp] = std::max<RATETYPE>(_parameter_set._delay_sets[ dreact_##name ], temp_delays[sp]); \
  } \
  for (int in = 0; in < _model.reaction_##name.getNumFactors(); in++) { \
    specie_id sp = _model.reaction_##name.getFactors()[in]; \
    temp_delays[sp] = std::max<RATETYPE>(_parameter_set._delay_sets[ dreact_##name ], temp_delays[sp]); \
  }
#include "reactions_list.hpp"
#undef REACTION
#undef DELAY_REACTION
    for (int s = 0; s < NUM_SPECIES; s++) {
        max_delays[s] = temp_delays[s]/_step_size;
    }
}
