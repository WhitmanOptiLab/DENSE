#ifndef SIM_DETERM_BABY_CL_CUDA_HPP
#define SIM_DETERM_BABY_CL_CUDA_HPP
#include "util/common_utils.hpp"
#include "core/specie.hpp"
#include "core/model.hpp"
#include "baby_cl.hpp"
#include <cuda.h>

#include <cstddef>

class baby_cl_cuda : public baby_cl {
  public:
    baby_cl_cuda(simulation_determ& sim) : baby_cl(sim) { }

    baby_cl_cuda(int length, int width, simulation_determ& sim) : baby_cl(length,width,sim) {}

    ~baby_cl_cuda() {
      dealloc_array();
    }

    void initialize();


protected:
    void dealloc_array(){
        (cudaFree(_array));
        _array = NULL;
    }

    void allocate_array(){
        if (_total_length >0){
            int size = _total_length * sizeof(RATETYPE);
            (cudaMallocManaged((void**)&_array, size));
            //_array= new RATETYPE[_total_length];
            //if (_array == NULL){std::cout<<"ERROR"<<std::endl; exit(EXIT_MEMORY_ERROR);}
        }
        else{
            _array= NULL;
        }
    }

};


#endif
