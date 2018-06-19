#ifndef SIM_DETERM_DENSE_CL_HPP
#define SIM_DETERM_DENSE_CL_HPP
#include "utility/common_utils.hpp"
#include "core/specie.hpp"
#include "core/model.hpp"

#include <cstddef>

class simulation;


class dense_cl {
  protected:
    const simulation& _sim;
    int   _length, _width,_total_length;
    Real *_array;

    int _max_delay = 0;
    int _specie_size = 0;

  public:
    class cell{
    public:

        IF_CUDA(__host__ __device__)
        cell(Real *row): _array(row) {}

        IF_CUDA(__host__ __device__)
        Real& operator[](int k){
            return _array[k];
        }

        IF_CUDA(__host__ __device__)
        const Real& operator[](int k) const {
            return _array[k];
        }
        Real *_array;
    };


    class timespan{
    public:
        IF_CUDA(__host__ __device__)
        timespan(Real *plane,int width, int pos): _array(plane), _width(width),_pos(pos) {};

        IF_CUDA(__host__ __device__)
        cell operator[](int j) {
            j = (j + _pos) % _pos;
            cell temp(_array+_width*j);
            return temp;
        }

        IF_CUDA(__host__ __device__)
        const cell operator[](int j) const{
            j = (j + _pos) % _pos;
            cell temp(_array+_width*j);
            return temp;
        }
        Real *_array;
        int _width;
        int _pos;
    };

    dense_cl(simulation& sim)
    :_width(0), _total_length(0),_sim(sim){
        allocate_array();
    }

    dense_cl(int length, int width, simulation& sim)
    :_width(width),_total_length(0),_sim(sim) {
        allocate_array();
    }

    void initialize();
    void reset(){
        for (int i = 0; i < _total_length; i++) {
            _array[i] = 0.0; // Initialize every concentration level at every time step for every cell to 0
        }
    }


    IF_CUDA(__host__ __device__)
    timespan operator[](int i) {
        return timespan(_array+(i*_sim._cells_total), _width, _max_delay);
    }

    IF_CUDA(__host__ __device__)
    const timespan operator[](int i) const {
        return timespan(_array+(i*_sim._cells_total), _width, _max_delay);
    }



    int width() const {return _width;}
    int total_length() const {return _total_length;}
    ~dense_cl() {
      dealloc_array();
    }


  protected:

    void dealloc_array() {
      delete[] _array;
      _array = nullptr;
    }

    void allocate_array() {
      _array = _total_length > 0 ? new Real[_total_length] : nullptr;
    }

};

#endif
