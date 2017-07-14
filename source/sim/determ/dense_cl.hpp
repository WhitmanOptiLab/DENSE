#ifndef SIM_DETERM_DENSE_CL_HPP
#define SIM_DETERM_DENSE_CL_HPP
#include "util/common_utils.hpp"
#include "core/specie.hpp"
#include "core/model.hpp"

using namespace std;

#include <cstddef>

class simulation;


class dense_cl {
  protected:
    const simulation& _sim;
    int   _length, _width,_total_length;
    RATETYPE *_array;

    int _max_delay = 0;
    int _specie_size = 0;

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
    
    dense_cl(simulation& sim)
    :_width(0), _total_length(0),_sim(sim){
        allocate_array();
    }
    
    dense_cl(int length, int width, simulation& sim)
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
    timespan operator[](int i) {
        return timespan(_array+(i*_sim._cells_total), _width, _max_delay);
    }

    CPUGPU_FUNC
    const timespan operator[](int i) const {
        return timespan(_array+(i*_sim._cells_total), _width, _max_delay);
    }
    
    
    
    int width() const {return _width;}
    int total_length() const {return _total_length;}
    ~dense_cl() {
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
        }
        else{
            _array= NULL;
        }
    }
    
};

#endif

