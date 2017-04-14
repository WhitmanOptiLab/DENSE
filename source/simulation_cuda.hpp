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
    class Context: public simulation::Context{
    public:
        typedef CPUGPU_TempArray<RATETYPE, NUM_SPECIES> SpecieRates;
        CPUGPU_FUNC
        Context(simulation_cuda& sim, int cell) : simulation::Context(sim, cell) { }
        CPUGPU_FUNC
        RATETYPE getCon(specie_id sp, int delay = 1) const {
            int modified_step = _simulation._baby_j[sp] + 1 - delay;
            return static_cast<simulation_cuda&>(_simulation)._baby_cl_cuda[sp][modified_step][_cell];
        }
    };
    baby_cl_cuda _baby_cl_cuda;
    CPUGPU_TempArray<int, 6>* _old_neighbors;
    RATETYPE* _old_rates;
    RATETYPE* _old_delays;
    RATETYPE* _old_crits;
    void initialize();
    void simulate_cuda(RATETYPE sim_time);
    simulation_cuda(const model& m, const param_set& ps, int cells_total, int width_total, RATETYPE step_size) :
        simulation(m,ps,cells_total,width_total,step_size), _baby_cl_cuda(*this) {
          _old_neighbors = _neighbors;
          cudaMallocManaged(&_neighbors, sizeof(CPUGPU_TempArray<int, 6>)*_cells_total);
          _old_rates = _rates._array;
          cudaMallocManaged(&(_rates._array), sizeof(RATETYPE)*_cells_total);
          _old_delays = _delays._array;
          cudaMallocManaged(&(_delays._array), sizeof(RATETYPE)*_cells_total);
          _old_crits = _critValues._array;
          cudaMallocManaged(&(_critValues._array), sizeof(RATETYPE)*_cells_total);
        }
    ~simulation_cuda() {
      cudaFree(_delays._array);
      _delays._array = _old_delays;
      cudaFree(_rates._array);
      _rates._array = _old_rates;
      cudaFree(_critValues._array);
      _critValues._array = _old_crits;
      cudaFree(_neighbors);
      _neighbors = _old_neighbors;
    }
};
#endif

