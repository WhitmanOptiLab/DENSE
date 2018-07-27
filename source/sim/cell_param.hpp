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
    Natural cell_count_ = {};
    Natural simulation_width_ = {};
    T *_array;

    IF_CUDA(__host__ __device__)
    cell_param(Natural width_total, Natural cells_total);

    ~cell_param() {
      delete[] _array;
    }

    IF_CUDA(__host__ __device__)
    T* operator[](int i){
      return &_array[cell_count_ * i];
    }

    IF_CUDA(__host__ __device__)
    T const* operator[](int i) const{
      return &_array[cell_count_ * i];
    }

    void initialize_params(Parameter_Set const& ps, Real normfactor = 1.0, Real* factors_perturb = nullptr, Real** factors_gradient = nullptr);

    int height() const {
      return _height;
    }

    int width() const {
      return cell_count_;
    }

    static inline T random_perturbation (T perturb) {
      return perturb == 0 ? 1 : random_rate(1 - perturb, 1 + perturb);
    }
    static T random_rate(T minimum, T maximum) {
      return minimum + (maximum - minimum) * rand() / (RAND_MAX + 1.0);
    }

};

}

#endif
