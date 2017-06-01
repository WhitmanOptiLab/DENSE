#ifndef BABY_CL_HPP
#define BABY_CL_HPP
#include "specie.hpp"
#include "model.hpp"
#define WRAP(x, y) ((x) + (y)) % (y)
#define MAX(x, y) ((x) < (y) ? (y) : (x))
using namespace std;


// Comment/Uncomment this next line to enable/disable 3D array test data structure.
// To rid these files of the SPARSE_STRUCT feature entirely, use a text editor's regex find and replace feature to replace "#else\n.*#endif\n", "#endif\n", and "#ifndef SPARSE_STRUCT" with blank lines
//#define SPARSE_STRUCT


#include <cstddef>

class simulation;


class baby_cl {
    //FIXME - want to make this private at some point
  protected:
    const simulation& _sim;
    //const model& _model;
    int   _length, _width,_total_length;
    RATETYPE *_array;
//    RATETYPE *_darray;

#ifndef SPARSE_STRUCT
    int _position[NUM_SPECIES];
    int _specie_size[NUM_SPECIES];
#else
    int _max_delay = 0;
    int _specie_size = 0;
#endif

  public:
    class cell{
    public:
        
        CPUGPU_FUNC
        cell(RATETYPE *row): _array(row) {}
        
        CPUGPU_FUNC
        RATETYPE& operator[](int k){
            return _array[k];
        }
        
        CPUGPU_FUNC
        const RATETYPE& operator[](int k) const {
            return _array[k];
        }
        RATETYPE *_array;
    };
    
    
    class timespan{
    public:
        CPUGPU_FUNC
        timespan(RATETYPE *plane,int width, int pos): _array(plane), _width(width),_pos(pos) {};
        
        CPUGPU_FUNC
        cell operator[](int j) {
            j = WRAP(j, _pos);
            cell temp(_array+_width*j);
            return temp;
        }
        
        CPUGPU_FUNC
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
    :_width(0), _total_length(0),_sim(sim){
        allocate_array();
    }
    
    baby_cl(int length, int width, simulation& sim)
    :_width(width),_total_length(0),_sim(sim){
        allocate_array();
    }
    
    void initialize();
    void reset(){
        for (int i = 0; i < _total_length; i++) {
            _array[i] = 0.0; // Initialize every concentration level at every time step for every cell to 0
        }
    }
    
    
    CPUGPU_FUNC
    timespan operator[](int i);
    
    CPUGPU_FUNC
    const timespan operator[](int i) const;
    
    
    int width() const {return _width;}
    int total_length() const {return _total_length;}
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
            _array= new RATETYPE[_total_length];
            //if (_array == NULL){std::cout<<"ERROR"<<std::endl; exit(EXIT_MEMORY_ERROR);}
        }
        else{
            _array= NULL;
        }
    }
    
};


#endif

