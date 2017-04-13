#ifndef CONCENTRATION_LEVEL_HPP
#define CONCENTRATION_LEVEL_HPP
#include "specie.hpp"
#include "model.hpp"
using namespace std;

class simulation;
template<class TYPE, int SIZE>
class CPUGPU_TempArray {
  TYPE array[SIZE];
 public:
  CPUGPU_FUNC
  TYPE& operator[](int i) { return array[i]; }
  CPUGPU_FUNC
  const TYPE& operator[](int i) const { return array[i]; }
};


class Concentration_level {
    //FIXME - want to make this private at some point
public:
    const simulation& _sim;
    //const model& _model;
    int   _height,_length, _width;
    bool _cuda;
    RATETYPE *_array;
    RATETYPE *_darray;
    
    
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
        timespan(RATETYPE *plane,int width): _array(plane), _width(width) {};
        cell operator[](int j) {
            cell temp(_array+_width*j);
            return temp;
        }
        
        const cell operator[](int j) const{
            cell temp(_array+_width*j);
            return temp;
        }
        RATETYPE *_array;
        int _width;
    };
    
    Concentration_level(simulation& sim)
    :_height(NUM_SPECIES),_width(0),_length(0), _cuda(false),_sim(sim){
        allocate_array();
    }
    
    Concentration_level(int height, int length, int width, simulation& sim)
    :_height(height),_length(length),_width(width),_cuda(false),_sim(sim){
        allocate_array();
    }
/*
#if 0
    concentration_level(const concentration_level& other)
    :_height(other._height),_length(other._length),_width(other._width),_cuda(other._cuda){
        allocate_array();
        for (int he=0; he< _height; he++){
            for (int le=0; le< _length; le++){
                for (int wi=0; wi< _width; wi++){
                    _array[he*_length*_width+le*_width+wi]=other._array[he*_length*_width+le*_width+wi];
                    _darray[he*_length*_width+le*_width+wi]=other._darray[he*_length*_width+le*_width+wi];
                }
            }
        }
    }
#endif
*/
    
    void initialize(int height, int length, int width){
        dealloc_array();
        _width=width;
        _height=height;
        //_length=length;
        _cuda=false;
        allocate_array();
        reset();
    }
    
    
    
    //RATETYPE calc_delay(int relatedReactions[]);
    
    void createArrary(){
        
    }
    void reset(){
        for (int i = 0; i < _height; i++) {
            for (int j = 0; j < _length; j++) {
                for (int k = 0; k < _width; k++) {
                    _array[i*_length*_width+j*_width+k] = 0; // Initialize every concentration level at every time step for every cell to 0
                    
                }
            }
        }
    }
    
    /*
     concentration_level& operator=(const concentration_level& other){
     dealloc_array();
     _height=other._height;
     _length=other._length;
     _width=other._width;
     allocate_array();
     for (int he; he< _height; he++){
     for (int le; le< _length; le++){
     for (int wi; wi< _width; wi++){
     _array[he*_length*_width+le*_width+wi]=other._array[he*_length*_width+le*_width+wi];
     }
     }
     }
     return *this;
     }
     */
    timespan operator[](int i){
        if (_cuda){
            timespan temp(_darray+_length*_width*i, _width);
            return temp;
        }
        else{
            timespan temp(_array+_length*_width*i, _width);
            return temp;
        }
    }
    
    const timespan operator[](int i) const{
        if (_cuda){
            timespan temp(_darray+_length*_width*i, _width);
            return temp;
        }
        else{
            timespan temp(_array+_length*_width*i, _width);
            return temp;
        }
    }

    int height() const {return _height;}
    int length() const {return _length;}
    int width() const {return _width;}
    bool getStatus() { return _cuda; }
    
    
protected:
    void dealloc_array(){
        if (_array){
            delete[] _array;
        }
        _array= NULL;
    }
    
    void allocate_array(){
        if (_width*_length*_height >0){
            _array= new RATETYPE[_height*_length*_width];
            //if (_array == NULL){std::cout<<"ERROR"<<std::endl; exit(EXIT_MEMORY_ERROR);}
        }
        else{
            _array= NULL;
        }
    }
    
};


#endif

