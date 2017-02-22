#ifndef BABY_CL_HPP
#define BABY_CL_HPP
#include "specie.hpp"
#include "model.hpp"
#define WRAP(x, y) ((x) + (y)) % (y)
#define MAX(x, y) ((x) < (y) ? (y) : (x))
using namespace std;


class simulation;

class baby_cl {
    //FIXME - want to make this private at some point
public:
    const simulation& _sim;
    //const model& _model;
    int   _height,_length, _width,_total_length;
    bool _cuda;
    RATETYPE *_array;
    RATETYPE *_darray;
    int _position[NUM_SPECIES];
    int _delay_size[NUM_SPECIES];
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
    
    baby_cl(simulation& sim)
    :_height(NUM_SPECIES),_length(), _cuda(false),_sim(sim){
        allocate_array();
    }
    
    baby_cl(int helight, int length, int width, simulation& sim)
    :_height(),_length(),_width(),_cuda(false),_sim(sim){
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
    /*
    void initialize(int height, int length, int width, bool delay){
        dealloc_array();
        _width=width;
        _height=height;
        //_length=length;
        _cuda=false;
        allocate_array();
        reset();
    }
    */
    
    
    RATETYPE calc_delay(int relatedReactions[]);
    void fill_position();
    void initialize();
    void reset(){
        for (int i = 0; i < _total_length; i++) {
                    _array[i] = 0; // Initialize every concentration level at every time step for every cell to 0
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
            timespan temp(_darray+_length*_width*i, _width, _delay_size[i] );
            return temp;
        }
        else{
            //int pos =
            timespan temp(_array+_position[i], _width, _delay_size[i]);
            return temp;
        }
    }
    
    const timespan operator[](int i) const{
        if (_cuda){
            
            timespan temp(_darray+_position[i], _width, _delay_size[i]);
            return temp;
        }
        else{
            
            timespan temp(_array+_position[i], _width, _delay_size[i]);
            return temp;
        }
    }
    
    int height() const {return _height;}
    int length() const {return _length;}
    int width() const {return _width;}
    int total_length() const {return _total_length;}
    bool getStatus() { return _cuda; }
    
    
protected:
    void dealloc_array(){
        if (_array){
            delete[] _array;
        }
        _array= NULL;
    }
    
    void allocate_array(){
        if (_total_length >0){
            _array= new RATETYPE[_total_length];
            //if (_array == NULL){std::cout<<"ERROR"<<std::endl; exit(EXIT_MEMORY_ERROR);}
        }
        else{
            _array= NULL;
        }
    }
    
};


#endif

