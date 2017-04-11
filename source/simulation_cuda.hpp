#ifndef SIMULATION_CUDA_HPP
#define SIMULATION_CUDA_HPP

#include "param_set.hpp"
#include "model.hpp"
#include "cell_param.hpp"
#include "reaction.hpp"
#include "concentration_level.hpp"
#include "baby_cl_cuda.hpp"
#include "simulation.hpp"
#include <vector>
#include <array>
using namespace std;

/* simulation contains simulation data, partially taken from input_params and partially derived from other information
	notes:
 There should be only one instance of simulation at any time.
	todo:
 */


class simulation_cuda: public simulation {
  public:
    baby_cl_cuda _baby_cl_cuda;
    void initialize();
    __global__ void simulate_cuda();
    simulation_cuda(const model& m, const param_set& ps, int cells_total, int width_total, RATETYPE step_size) :
        simulation(m,ps,cells_total,width_total,step_size), _baby_cl_cuda(*this) {}
};
#endif

