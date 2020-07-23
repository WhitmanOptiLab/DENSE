#ifndef SIM_DETERM_BABY_CL_HPP
#define SIM_DETERM_BABY_CL_HPP

#include "utility/common_utils.hpp"
#include "core/specie.hpp"
#include "core/model.hpp"
#include <memory>

namespace dense {

// Simulation_History
class baby_cl {

  private:

    Natural _position[NUM_SPECIES];
    int _specie_size[NUM_SPECIES];
    int _j[NUM_SPECIES] = {};
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
        unsigned sum = 0;
        for (int i = 0; i < NUM_SPECIES; i++) {
            _position[i] = sum * _width; //for each specie
            _specie_size[i] = (sim.max_delays[i] / sim._step_size) + sim._num_history_steps + 2;
            sum += _specie_size[i];
        }
        _total_length = sum * NUM_SPECIES * _width;
        _array = decltype(_array){new Real[_total_length + sim.num_growth_cells()]()};
    }

    int get_species_size(int species){
      return _specie_size[species];
    }

public:
    Real* row_at(int species, int j) {
      j = (j == 0) ? _j[species] : wrap(_j[species] + j, _specie_size[species]);
      return &_array[_position[species] + _width * j];
    }

    Real const* row_at(int species, int j) const {
      j = (j == 0) ? _j[species] : wrap(_j[species] + j, _specie_size[species]);
      return &_array[_position[species] + _width * j];
    }

    CUDA_AGNOSTIC
    void advance() {
      for (int i = 0; i < NUM_SPECIES; i++) {
        _j[i] = wrap(_j[i]+1, _specie_size[i]);
      }
    }
};

}



#endif
