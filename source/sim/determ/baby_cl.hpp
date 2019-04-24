#ifndef SIM_DETERM_BABY_CL_HPP
#define SIM_DETERM_BABY_CL_HPP

#include "utility/common_utils.hpp"
#include "core/specie.hpp"
#include "core/model.hpp"
#include <memory>

namespace dense {

class Deterministic_Simulation;
// Simulation_History
class baby_cl {

  private:

    int _position[NUM_SPECIES];
    int _specie_size[NUM_SPECIES];
    int _j[NUM_SPECIES] = {};
    int _width;
    unsigned _total_length;
    std::unique_ptr<Real[]> _array;

  public:

    template <typename NumericT>
    CUDA_AGNOSTIC
    static NumericT wrap (NumericT x, NumericT y) {
      return (x + y) % y;
    }

    baby_cl(Deterministic_Simulation& sim);

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
