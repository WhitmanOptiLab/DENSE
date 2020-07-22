#ifndef SIM_DETERM_BABY_CL_HPP
#define SIM_DETERM_BABY_CL_HPP

#include "utility/common_utils.hpp"
#include "core/specie.hpp"
#include "core/model.hpp"
#include <memory>

namespace dense {
/*
class Deterministic_Simulation;
class Simpson_Simulation;
class Trapezoid_Simulation;
class Average_Simulation;
*/
// Simulation_History
class baby_cl {

  private:

    //Natural _position[NUM_SPECIES];
    int _specie_size;
    int _j = 0;
    Natural _width;
    unsigned _total_length;
    std::unique_ptr<Real[]> _array;

  public:

    template <typename NumericT>
    CUDA_AGNOSTIC
    static NumericT wrap (NumericT x, NumericT y) {
      return (x + y) % y;
    }

    template<class Numerical_Simulation>
    baby_cl(Numerical_Simulation& sim) : _width{sim.cell_count()} {
        int max_delay = 0;
        for (int i = 0; i < NUM_SPECIES; i++) {
            if (sim.max_delays[i] >= max_delay) {
              _specie_size = (sim.max_delays[i] / sim._step_size) + sim._num_history_steps;
              max_delay = sim.max_delays[i];
            }
        }
        _total_length = _specie_size * NUM_SPECIES * _width;
        _array = decltype(_array){new Real[_total_length + sim.num_growth_cells()]()};
    }

    int get_species_size(int) {
      return _specie_size;
    }

public:
    Real* row_at(int species, int j) {
      j = (j == 0) ? _j : wrap(_j + j, _specie_size);
      return &_array[_width*(species*_specie_size + j)];
    }

    Real const* row_at(int species, int j) const {
      j = (j == 0) ? _j : wrap(_j + j, _specie_size);
      return &_array[_width*(species*_specie_size + j)];
    }

    CUDA_AGNOSTIC
    void advance() {
      _j = wrap(_j+1, _specie_size);
    }
};

}



#endif
