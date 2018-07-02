#ifndef SIM_DETERM_DETERM_CUDA_HPP
#define SIM_DETERM_DETERM_CUDA_HPP

#include "core/parameter_set.hpp"
#include "core/model.hpp"
#include "sim/cell_param.hpp"
#include "core/reaction.hpp"
#include "baby_cl_cuda.hpp"
#include "determ.hpp"
#include <vector>
#include <array>
#include <iostream>

#define check(RESULT) do {\
  check(RESULT, __FILE__, __LINE__);\
} while(0)

inline void (check)(cudaError code, const char *file, unsigned line) {
  if (code != cudaError{}) {
    std::cerr << file << ':' << line << ": " << cudaGetErrorString(code) << '\n';
    exit(-1);
  }
}

class simulation_cuda: public Deterministic_Simulation {
  public:
    class Context : public dense::Context {
        //FIXME - want to make this private at some point
      protected:
        int _cell;
        simulation_cuda& _simulation;
        double _avg;

      public:
        typedef CUDA_Array<Real, NUM_SPECIES> SpecieRates;
        IF_CUDA(__host__ __device__)
        Context(simulation_cuda& sim, int cell) : _simulation(sim),_cell(cell) { }
        IF_CUDA(__host__ __device__)
        Real calculateNeighborAvg(specie_id sp, int delay = 1) const;
        IF_CUDA(__host__ __device__)
        void updateCon(const SpecieRates& rates);
        IF_CUDA(__host__ __device__)
        const SpecieRates calculateRatesOfChange();
        IF_CUDA(__host__ __device__)
        virtual Real getCon(specie_id sp) const final {
          return getCon(sp, 1);
        }
        IF_CUDA(__host__ __device__)
        Real getCon(specie_id sp, int delay) const {
            return _simulation._baby_cl_cuda[sp][1 - delay][_cell];
        }
        IF_CUDA(__host__ __device__)
        Real getCritVal(critspecie_id rcritsp) const {
            return _simulation._cellParams[rcritsp + NUM_REACTIONS + NUM_DELAY_REACTIONS][_cell];
        }
        IF_CUDA(__host__ __device__)
        Real getRate(reaction_id reaction) const {
            return _simulation._cellParams[reaction][_cell];
        }
        IF_CUDA(__host__ __device__)
        int getDelay(delay_reaction_id delay_reaction) const{
            return _simulation._cellParams[delay_reaction + NUM_REACTIONS][_cell];
        }
        IF_CUDA(__host__ __device__)
        virtual void advance() final { ++_cell; }
        IF_CUDA(__host__ __device__)
        virtual void set(int c) final {_cell = c;}
        IF_CUDA(__host__ __device__)
        virtual bool isValid() const final { return _cell >= 0 && _cell < _simulation._cells_total; }
    };

    baby_cl_cuda _baby_cl_cuda;
    CUDA_Array<int, 6>* _old_neighbors;
    int* _old_numNeighbors;
    Real* _old_cellParams;
    int* _old_intDelays;
    Real* _old_crits;
    void initialize();

    IF_CUDA(__host__ __device__)
    void execute_one(int k) {
        // Iterate through each extant cell or context
        //if (_width_current == _width_total || k % _width_total <= 10) { // Compute only existing (i.e. already grown)cells
            // Calculate the cell indices at the start of each mRNA and protein's dela
            Context c(*this, k);

            // Perform biological calculations
            //_baby_cl_cuda[ph11][0][0] = 5.0; works
            //c.calculateRatesOfChange();
            update_concentrations(k, c.calculateRatesOfChange());
        //}
        if (k==0){
            _j++;
            _baby_cl_cuda.advance();
        }
    }

    void simulate_cuda();
    simulation_cuda(const dense::model& m, const Parameter_Set& ps, int cells_total, int width_total, Real step_size, Real analysis_interval, Real sim_time) :
        Deterministic_Simulation(m,ps,nullptr,nullptr, cells_total, width_total,step_size, analysis_interval, sim_time), _baby_cl_cuda(*this) {
          _old_neighbors = _neighbors;
          check(cudaMallocManaged(&_neighbors, sizeof(CUDA_Array<int, 6>)*_cells_total));
          _old_numNeighbors = _numNeighbors;
          check(cudaMallocManaged(&_numNeighbors, sizeof(int)*_cells_total));
          _old_cellParams = _cellParams._array;
          check(cudaMallocManaged(&(_cellParams._array), sizeof(Real)*_cells_total*NUM_PARAMS));
          _old_intDelays = _intDelays._array;
          check(cudaMallocManaged(&(_intDelays._array), sizeof(Real)*_cells_total*NUM_DELAY_REACTIONS));
        }
    ~simulation_cuda() {
      cudaFree(_intDelays._array);
      _intDelays._array = _old_intDelays;
      cudaFree(_cellParams._array);
      _cellParams._array = _old_cellParams;
      cudaFree(_neighbors);
      _neighbors = _old_neighbors;
      cudaFree(_numNeighbors);
      _numNeighbors = _old_numNeighbors;
    }
};
#endif
