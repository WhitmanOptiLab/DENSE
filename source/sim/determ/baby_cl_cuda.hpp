#ifndef SIM_DETERM_BABY_CL_CUDA_HPP
#define SIM_DETERM_BABY_CL_CUDA_HPP
#include "utility/common_utils.hpp"
#include "core/specie.hpp"
#include "core/model.hpp"
#include "baby_cl.hpp"
#include <cuda.h>

#include <cstddef>

class baby_cl_cuda : public baby_cl {
  public:
    baby_cl_cuda(Deterministic_Simulation& sim) : baby_cl(sim) { }

    baby_cl_cuda(int length, int width, Deterministic_Simulation& sim) : baby_cl(length,width,sim) {}

    ~baby_cl_cuda() {
      dealloc_array();
    }

    void initialize();


protected:
    void dealloc_array(){
        (cudaFree(_array));
        _array = nullptr;
    }

    void allocate_array(){
        if (_total_length >0){
            int size = _total_length * sizeof(Real);
            (cudaMallocManaged((void**)&_array, size));
            //_array= new Real[_total_length];
            //if (_array == nullptr){std::cout<<"ERROR"<<std::endl; exit(EXIT_MEMORY_ERROR);}
        }
        else{
            _array= nullptr;
        }
    }

};


#endif
