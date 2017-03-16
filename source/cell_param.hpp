#ifndef CELL_PARAM_HPP
#define CELL_PARAM_HPP

#include <stdlib.h>
//#include <cuda_runtime_api.h>
//#include <cuda.h>
#include <iostream>
//#include "simulation.hpp"
#include "reaction.hpp"
#include "specie.hpp"

/*
#ifdef __CUDACC__
#define CPUGPU_FUNC __host__ __device__
#else
#define CPUGPU_FUNC
#endif
*/
using namespace std;
class simulation;

template<int N>
class cell_param {
    //FIXME - want to make this private at some point
public:
    int   _height, _width;
    bool _cuda;
    RATETYPE *_array;
    //int _delay_size[NUM_REACTIONS];
    RATETYPE *_darray;
    const simulation& _sim;
    
    
    class cell{
    public:
        cell(RATETYPE *row): _array(row) {}
        RATETYPE& operator[](int k){
            return _array[k];
        }
        const RATETYPE& operator[](int k) const {
            return _array[k];
        }
        RATETYPE *_array;
    };
    
    
    cell_param(const simulation& sim)
    :_height(N),_sim(sim),_cuda(false){
        //_delay_size = int
        allocate_array();
    }
    
    
    cell operator[](int i){
        if (_cuda){
            cell temp(_darray+_width*i);
            return temp;
        }
        else{
            cell temp(_array+_width*i);
            return temp;
        }
    }
    
    const cell operator[](int i) const{
        if (_cuda){
            cell temp(_darray+_width*i);
            return temp;
        }
        else{
            cell temp(_array+_width*i);
            return temp;
        }
    }
    
    void update_rates(const RATETYPE param_data[]);
    int height() const {return _height;}
    int width() const {return _width;}
    inline RATETYPE random_perturbation (RATETYPE perturb) {
        return random_rate(pair<RATETYPE, RATETYPE>(1 - perturb, 1 + perturb));
    }
    RATETYPE random_rate(pair<RATETYPE, RATETYPE> range) {
        return range.first + (range.second - range.first) * rand() / (RAND_MAX + 1.0);
    }
    void initialize();
//protected:
    void dealloc_array(){
        if (_array){
            delete[] _array;
        }
        _array= NULL;
    }
    
    void allocate_array(){
        if (_width*_height >0){
            _array= new RATETYPE[_height*_width];
            //if (_array == NULL){std::cout<<"ERROR"<<std::endl; exit(EXIT_MEMORY_ERROR);}
        }
        else{
            _array= NULL;
        }
    }

};


#endif
