#ifndef SIM_CELL_PARAM_HPP
#define SIM_CELL_PARAM_HPP

//#include <cuda_runtime_api.h>
//#include <cuda.h>
//#include "simulation.hpp"
#include "utility/common_utils.hpp"
#include "core/reaction.hpp"
#include "core/specie.hpp"
#include "core/parameter_set.hpp"

namespace dense {

class Simulation;

template<int N, class T = Real>
class cell_param {
static constexpr Natural _height = N;
    //FIXME - want to make this private at some point
public:

    cell_param () noexcept = default;

    Natural cell_count_ = {};
    Natural simulation_width_ = {};
    T *_array = {};

    CUDA_AGNOSTIC
    cell_param(Natural width_total, Natural cells_total);

    CUDA_AGNOSTIC
    ~cell_param() {
      delete[] _array;
    }

    CUDA_AGNOSTIC
    T* operator[](int i){
      return &_array[cell_count_ * i];
    }

    CUDA_AGNOSTIC
    T const* operator[](int i) const{
      return &_array[cell_count_ * i];
    }

    CUDA_AGNOSTIC
    int height() const {
      return _height;
    }

    CUDA_AGNOSTIC
    int width() const {
      return cell_count_;
    }
};

}

template
class dense::cell_param<NUM_PARAMS>;

template <typename T>
CUDA_AGNOSTIC
inline T random_rate(T minimum, T maximum) {
  return minimum + (maximum - minimum) * rand() / (RAND_MAX + 1.0);
}

template <typename T>
CUDA_AGNOSTIC
inline T random_perturbation (T perturb) {
  return perturb == 0 ? 1 : random_rate(1 - perturb, 1 + perturb);
}

#include "cell_param.ipp"

#endif
