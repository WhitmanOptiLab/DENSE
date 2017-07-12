#ifndef SIM_DETERM_BABY_CL_HPP
#define SIM_DETERM_BABY_CL_HPP
#include "core/specie.hpp"
#include "core/model.hpp"
#define WRAP(x, y) ((x) + (y)) % (y)
#define MAX(x, y) ((x) < (y) ? (y) : (x))
using namespace std;

#include <cstddef>

class simulation_determ;


class baby_cl {
    //FIXME - want to make this private at some point
  protected:
    const simulation_determ& _sim;
    //const model& _model;
    int   _length, _width,_total_length;
    RATETYPE *_array;
//    RATETYPE *_darray;

    int _position[NUM_SPECIES];
    int _specie_size[NUM_SPECIES];

    int _j[NUM_SPECIES];

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
        timespan(RATETYPE *plane,int width, int pos, int hist_len): _array(plane), _width(width),_pos(pos),_hist_len(hist_len) {};
        
        CPUGPU_FUNC
        cell operator[](int j) {
            j = (j == 0) ? _pos : WRAP(_pos + j, _hist_len);
            cell temp(_array+_width*j);
            return temp;
        }
        
        CPUGPU_FUNC
        const cell operator[](int j) const{
            j = (j == 0) ? _pos : WRAP(_pos + j, _hist_len);
            cell temp(_array+_width*j);
            return temp;
        }
        RATETYPE *_array;
        int _width;
        int _pos;
        int _hist_len;
    };
    
    baby_cl(simulation_determ& sim)
    :_width(0), _total_length(0),_sim(sim){
        allocate_array();
        for (int i = 0; i < NUM_SPECIES; i++) {
          _j[i] = 0;
        }
    }
    
    baby_cl(int length, int width, simulation_determ& sim)
    :_width(width),_total_length(0),_sim(sim){
        allocate_array();
        for (int i = 0; i < NUM_SPECIES; i++) {
          _j[i] = 0;
        }
    }
    
    void initialize();
    void reset(){
        for (int i = 0; i < _total_length; i++) {
            _array[i] = 0.0; // Initialize every concentration level at every time step for every cell to 0
        }
        for (int i = 0; i < NUM_SPECIES; i++) {
          _j[i] = 0;
        }
    }
    
    
    CPUGPU_FUNC
    timespan operator[](int i) {
        return timespan(_array+_position[i], _width, _j[i], _specie_size[i]);
    }

    CPUGPU_FUNC
    const timespan operator[](int i) const {
        return timespan(_array+_position[i], _width, _j[i], _specie_size[i]);
    }
    
    CPUGPU_FUNC
    void advance() { 
      for (int i = 0; i < NUM_SPECIES; i++) {
        _j[i] = WRAP(_j[i]+1, _specie_size[i]);
      }
    }
    
    
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

