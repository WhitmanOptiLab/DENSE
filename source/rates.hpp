#ifndef RATES_HPP
#define RATES_HPP

#include <stdlib.h>
//#include <cuda_runtime_api.h>
//#include <cuda.h>
#include <iostream>
//#include "simulation.hpp"
#include "reaction.hpp"
#include "specie.hpp"
#include "reaction.hpp"
/*
#ifdef __CUDACC__
#define CPUGPU_FUNC __host__ __device__
#else
#define CPUGPU_FUNC
#endif
*/
using namespace std;
class simulation;

class Rates {
    //FIXME - want to make this private at some point
public:
    int   _height, _width;
    bool _cuda;
    RATETYPE *_array;
    int _delay_size[NUM_REACTIONS];
    RATETYPE *_darray;
    const simulation& _sim;
    
    //std::vector<E> rates_active;
    //std::vector<int> delay_size;
    //Rates(const simulation& _sim) : _simulation(_sim), rates_base(NUM_REACTIONS*cells), rates_active(NUM_REACTIONS*cells), delay_size(NUM_REACTIONS) { }
    
    
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
    
    
    Rates(const simulation& sim)
    :_height(NUM_SPECIES),_sim(sim),_cuda(false){
        //_delay_size = int
        allocate_array();
    }
    
    /*
    Rates(const array2D<RATETYPE>& other)
    :_height(other._height),_width(other._width){
        //allocate_array();
        _array=other._array;
        //_darray=other._darray;
    }*/
    
    /*
    void initialize(int height, int width){
        dealloc_array();
        _width=width;
        _height=height;
        //_cuda=false;
        allocate_array();
        reset();
    }
    
    void initialize_as_1(int height, int width){
        dealloc_array();
        _width=width;
        _height=height;
        _cuda=false;
        allocate_array();
        set_to_1();
    }
    
    
    void set_to_1(){
        for (int i = 0; i < _height; i++) {
            for (int k = 0; k < _width; k++) {
                _array[i*_width+k] = 1; // Initialize every concentration level at every time step for every cell to 0
                
            }
        }
    }
    
    void reset(){
        for (int i = 0; i < _height; i++) {
            for (int k = 0; k < _width; k++) {
                _array[i*_width+k] = 0; // Initialize every concentration level at every time step for every cell to 0
                
            }
        }
    }
    */
    
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
    
    void update_rates();
    int height() const {return _height;}
    int width() const {return _width;}
    inline RATETYPE random_perturbation (RATETYPE perturb) {
        return random_rate(pair<RATETYPE, RATETYPE>(1 - perturb, 1 + perturb));
    }
    RATETYPE random_rate(pair<RATETYPE, RATETYPE> range) {
        return range.first + (range.second - range.first) * rand() / (RAND_MAX + 1.0);
    }
protected:
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
