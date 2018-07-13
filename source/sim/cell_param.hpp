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
static constexpr dense::Natural _height = N;
    //FIXME - want to make this private at some point
public:
    Simulation const& _sim;
    dense::Natural _width;
    T *_array;
    bool _cuda;


    class cell{
    public:
        IF_CUDA(__host__ __device__)
        cell(T *row): _array(row) {}
        IF_CUDA(__host__ __device__)
        T& operator[](int k){
            return _array[k];
        }
        IF_CUDA(__host__ __device__)
        T const& operator[](int k) const {
            return _array[k];
        }
        T *_array;
    };

    IF_CUDA(__host__ __device__)
    cell_param(Simulation const& sim, dense::Natural ncells)
    :_sim(sim),_width(ncells),_cuda(false){
        allocate_array();
    }

    ~cell_param() {
      dealloc_array();
    }

    IF_CUDA(__host__ __device__)
    cell operator[](int i){
        return cell(_array + _width * i);
    }

    IF_CUDA(__host__ __device__)
    const cell operator[](int i) const{
        return cell(_array + _width * i);
    }

    void initialize_params(Parameter_Set const& ps, Real normfactor = 1.0, Real* factors_perturb = nullptr, Real** factors_gradient = nullptr);
    int height() const {return _height;}
    int width() const {return _width;}
    inline T random_perturbation (T perturb) {
      return random_rate(1 - perturb, 1 + perturb);
    }
    T random_rate(T minimum, T maximum) {
      return minimum + (maximum - minimum) * rand() / (RAND_MAX + 1.0);
    }
    void initialize();
//protected:
    IF_CUDA(__host__ __device__)
    void dealloc_array() {
      delete[] _array;
      _array = nullptr;
    }

    IF_CUDA(__host__ __device__)
    void allocate_array() {
      _array = _width > 0 && _height > 0 ? new T[_height * _width] : nullptr;
    }

};

}

#endif
