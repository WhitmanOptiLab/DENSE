#ifndef SIM_CELL_PARAM_HPP
#define SIM_CELL_PARAM_HPP

//#include <cuda_runtime_api.h>
//#include <cuda.h>
//#include "simulation.hpp"
#include "util/common_utils.hpp"
#include "core/reaction.hpp"
#include "core/specie.hpp"
#include "core/param_set.hpp"


class Simulation;

template<int N, class T = RATETYPE>
class cell_param {
    //FIXME - want to make this private at some point
public:
    Simulation const& _sim;
    int   _height, _width;
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
    cell_param(Simulation const& sim, int ncells)
    :_sim(sim),_height(N),_width(ncells),_cuda(false){
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

    void initialize_params(param_set const& ps, RATETYPE normfactor = 1.0);
    int height() const {return _height;}
    int width() const {return _width;}
    inline T random_perturbation (T perturb) {
        return random_rate(std::pair<T, T>(1 - perturb, 1 + perturb));
    }
    T random_rate(std::pair<T, T> range) {
        return range.first + (range.second - range.first) * rand() / (RAND_MAX + 1.0);
    }
    void initialize();
//protected:
    IF_CUDA(__host__ __device__)
    void dealloc_array(){
        if (_array){
            delete[] _array;
        }
        _array = nullptr;
    }

    IF_CUDA(__host__ __device__)
    void allocate_array(){
        if (_width * _height >0){
            _array = new T[_height * _width];
            //if (_array == nullptr){std::cout<<"ERROR"<<'\n'; exit(EXIT_MEMORY_ERROR);}
        }
        else {
            _array = nullptr;
        }
    }

};


#endif
