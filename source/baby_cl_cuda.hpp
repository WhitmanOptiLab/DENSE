#ifndef BABY_CL_CUDA_HPP
#define BABY_CL_CUDA_HPP
#include "specie.hpp"
#include "model.hpp"
#define WRAP(x, y) ((x) + (y)) % (y)
#define MAX(x, y) ((x) < (y) ? (y) : (x))
using namespace std;

#include <cstddef>

class simulation_cuda;

class baby_cl_cuda {
    //FIXME - want to make this private at some point
  private:
    const simulation& _sim;
    //const model& _model;
    int   _length, _width,_total_length;
    bool _cuda;
    RATETYPE *_array;
    RATETYPE *_darray;
    int _position[NUM_SPECIES];
    int _specie_size[NUM_SPECIES];
  public:
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
    
    
    class timespan{
    public:
        timespan(RATETYPE *plane,int width, int pos): _array(plane), _width(width),_pos(pos) {};
        cell operator[](int j) {
            j = WRAP(j, _pos);
            cell temp(_array+_width*j);
            return temp;
        }
        
        const cell operator[](int j) const{
            j = WRAP(j, _pos);
            cell temp(_array+_width*j);
            return temp;
        }
        RATETYPE *_array;
        int _width;
        int _pos;
    };
    
    baby_cl_cuda(simulation& sim)
    :_width(0), _total_length(0),_cuda(true),_sim(sim){
        allocate_array();
    }
    
    baby_cl_cuda(int length, int width, simulation& sim)
    :_width(width),_total_length(0),_cuda(true),_sim(sim){
        allocate_array();
    }
    
    void initialize();
    void reset(){
        for (int i = 0; i < _total_length; i++) {
                    _array[i] = 0.0; // Initialize every concentration level at every time step for every cell to 0
        }
    }
    
    timespan operator[](int i){
        if (_cuda){
            timespan temp(_array+_position[i], _width, _specie_size[i]);
            return temp;
        }
        else{
            //int pos =
            timespan temp(_array+_position[i], _width, _specie_size[i]);
            return temp;
        }
    }
    
    const timespan operator[](int i) const{
        if (_cuda){
            
            timespan temp(_darray+_position[i], _width, _specie_size[i]);
            return temp;
        }
        else{
            
            timespan temp(_array+_position[i], _width, _specie_size[i]);
            return temp;
        }
    }
    
    int width() const {return _width;}
    int total_length() const {return _total_length;}
    bool getStatus() { return _cuda; }
    ~baby_cl() {
      dealloc_array();
    }
    
    
protected:
    void dealloc_array(){
        if (_array){
            delete[] _array;
        }
        _array= NULL;
    }
    
    void allocate_array(){
        if (_total_length >0){
            int size = _total_length * sizeof(RATETYPE);
            CUDA_ERRCHK(cudaMalloc((void**)&_darray, size));
            //_array= new RATETYPE[_total_length];
            //if (_array == NULL){std::cout<<"ERROR"<<std::endl; exit(EXIT_MEMORY_ERROR);}
        }
        else{
            _array= NULL;
        }
    }
    
};


#endif

