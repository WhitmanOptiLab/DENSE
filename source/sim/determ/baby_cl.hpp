#ifndef SIM_DETERM_BABY_CL_HPP
#define SIM_DETERM_BABY_CL_HPP
#include "util/common_utils.hpp"
#include "core/specie.hpp"
#include "core/model.hpp"

using namespace std;

#include <cstddef>

class simulation_determ;


class baby_cl {
  protected:
    const simulation_determ& _sim;
    int   _length, _width,_total_length;
    RATETYPE *_array;

    int _position[NUM_SPECIES];
    int _specie_size[NUM_SPECIES];

    int _j[NUM_SPECIES];

  public:
    class cell{
    public:

        IF_CUDA(__host__ __device__)
        cell(RATETYPE *row): _array(row) {}

        IF_CUDA(__host__ __device__)
        RATETYPE& operator[](int k){
            return _array[k];
        }

        IF_CUDA(__host__ __device__)
        const RATETYPE& operator[](int k) const {
            return _array[k];
        }
        RATETYPE *_array;
    };

    template <typename NumericT>
    IF_CUDA(__host__ __device__)
    static NumericT wrap (NumericT x, NumericT y) {
      return (x + y) % y;
    }

    class timespan{
    public:
        IF_CUDA(__host__ __device__)
        timespan(RATETYPE *plane,int width, int pos, int hist_len): _array(plane), _width(width),_pos(pos),_hist_len(hist_len) {};

        IF_CUDA(__host__ __device__)
        cell operator[](int j) {
            j = (j == 0) ? _pos : wrap(_pos + j, _hist_len);
            cell temp(_array+_width*j);
            return temp;
        }

        IF_CUDA(__host__ __device__)
        const cell operator[](int j) const{
            j = (j == 0) ? _pos : wrap(_pos + j, _hist_len);
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


    IF_CUDA(__host__ __device__)
    timespan operator[](int i) {
        return timespan(_array+_position[i], _width, _j[i], _specie_size[i]);
    }

    IF_CUDA(__host__ __device__)
    const timespan operator[](int i) const {
        return timespan(_array+_position[i], _width, _j[i], _specie_size[i]);
    }

    IF_CUDA(__host__ __device__)
    void advance() {
      for (int i = 0; i < NUM_SPECIES; i++) {
        _j[i] = wrap(_j[i]+1, _specie_size[i]);
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
        }
        else{
            _array= NULL;
        }
    }

};



#endif
