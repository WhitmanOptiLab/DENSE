#ifndef SIM_CELL_PARAM_HPP
#define SIM_CELL_PARAM_HPP


#include <stdlib.h>
//#include <cuda_runtime_api.h>
//#include <cuda.h>
#include <iostream>
//#include "simulation.hpp"
#include "util/common_utils.hpp"
#include "core/reaction.hpp"
#include "core/specie.hpp"
#include "core/param_set.hpp"

using namespace std;
class simulation_base;

template<int N, class T=RATETYPE>
class cell_param {
    //FIXME - want to make this private at some point
public:
    int   _height, _width;
    bool _cuda;
    T *_array;
    simulation_base const& _sim;
    
    
    class cell{
    public:
        CPUGPU_FUNC
        cell(T *row): _array(row) {}
        CPUGPU_FUNC
        T& operator[](int k){
            return _array[k];
        }
        CPUGPU_FUNC
        T const& operator[](int k) const {
            return _array[k];
        }
        T *_array;
    };
    
    CPUGPU_FUNC
    cell_param(simulation_base const& sim, int ncells)
    :_height(N),_width(ncells),_sim(sim),_cuda(false){
        allocate_array();
    }

    ~cell_param() {
      dealloc_array();
    }
    
    CPUGPU_FUNC
    cell operator[](int i){
        return cell(_array+_width*i);
    }
    
    CPUGPU_FUNC
    const cell operator[](int i) const{
        return cell(_array+_width*i);
    }
    
    void initialize_params(param_set const& ps, RATETYPE normfactor = 1.0);
    int height() const {return _height;}
    int width() const {return _width;}
    inline T random_perturbation (T perturb) {
        return random_rate(pair<T, T>(1 - perturb, 1 + perturb));
    }
    T random_rate(pair<T, T> range) {
        return range.first + (range.second - range.first) * rand() / (RAND_MAX + 1.0);
    }
    void initialize();
//protected:
    CPUGPU_FUNC
    void dealloc_array(){
        if (_array){
            delete[] _array;
        }
        _array= NULL;
    }
    
    CPUGPU_FUNC
    void allocate_array(){
        if (_width*_height >0){
            _array= new T[_height*_width];
            //if (_array == NULL){std::cout<<"ERROR"<<'\n'; exit(EXIT_MEMORY_ERROR);}
        }
        else{
            _array= NULL;
        }
    }

};


#endif
