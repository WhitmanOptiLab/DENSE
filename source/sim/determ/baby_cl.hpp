#ifndef SIM_DETERM_BABY_CL_HPP
#define SIM_DETERM_BABY_CL_HPP

#include "utility/common_utils.hpp"
#include "core/specie.hpp"
#include "core/model.hpp"

namespace dense {

class Deterministic_Simulation;

class baby_cl {

  protected:

    int _position[NUM_SPECIES];
    int _specie_size[NUM_SPECIES];
    int _j[NUM_SPECIES];
    Deterministic_Simulation const& _sim;
    int _length, _width;
    unsigned _total_length;
    Real *_array;


  public:

    class cell {

      public:

        CUDA_HOST CUDA_DEVICE
        cell(Real *row): _array(row) {}

        CUDA_HOST CUDA_DEVICE
        Real& operator[](int k){
            return _array[k];
        }

        CUDA_HOST CUDA_DEVICE
        const Real& operator[](int k) const {
            return _array[k];
        }
        Real *_array;
    };

    template <typename NumericT>
    CUDA_HOST CUDA_DEVICE
    static NumericT wrap (NumericT x, NumericT y) {
      return (x + y) % y;
    }

    class timespan {

      public:

        CUDA_HOST CUDA_DEVICE
        timespan(Real *plane,int width, int pos, int hist_len): _array(plane), _width(width),_pos(pos),_hist_len(hist_len) {};

        CUDA_HOST CUDA_DEVICE
        cell operator[](int j) {
            j = (j == 0) ? _pos : wrap(_pos + j, _hist_len);
            cell temp(_array+_width*j);
            return temp;
        }

        CUDA_HOST CUDA_DEVICE
        const cell operator[](int j) const{
            j = (j == 0) ? _pos : wrap(_pos + j, _hist_len);
            cell temp(_array+_width*j);
            return temp;
        }

        Real *_array;
        int _width;
        int _pos;
        int _hist_len;
    };

    baby_cl(Deterministic_Simulation& sim)
    :_sim(sim), _width(0), _total_length(0) {
        allocate_array();
        for (int i = 0; i < NUM_SPECIES; i++) {
          _j[i] = 0;
        }
    }

    baby_cl(int length, int width, Deterministic_Simulation& sim)
    : _sim(sim), _width(width), _total_length(0) {
        allocate_array();
        for (int i = 0; i < NUM_SPECIES; i++) {
          _j[i] = 0;
        }
    }

    void initialize();
    void reset(){
        for (unsigned i = 0; i < _total_length; i++) {
            _array[i] = 0.0; // Initialize every concentration level at every time step for every cell to 0
        }
        for (int i = 0; i < NUM_SPECIES; i++) {
          _j[i] = 0;
        }
    }


    CUDA_HOST CUDA_DEVICE
    timespan operator[](int i) {
        return timespan(_array+_position[i], _width, _j[i], _specie_size[i]);
    }

    CUDA_HOST CUDA_DEVICE
    const timespan operator[](int i) const {
        return timespan(_array+_position[i], _width, _j[i], _specie_size[i]);
    }

    CUDA_HOST CUDA_DEVICE
    void advance() {
      for (int i = 0; i < NUM_SPECIES; i++) {
        _j[i] = wrap(_j[i]+1, _specie_size[i]);
      }
    }

    int width() const {
      return _width;
    }

    int total_length() const {
      return _total_length;
    }

    ~baby_cl() {
      dealloc_array();
    }

  protected:

    void dealloc_array() {
        delete[] _array;
        _array = nullptr;
    }

    void allocate_array() {
        _array = _total_length != 0 ? new Real[_total_length] : nullptr;
    }

};

}



#endif
