#Would prefer to make this 3.1, as it makes C++ dialect selection automatic.  
cmake_minimum_required (VERSION 2.8)

project (DDESimulatorTemplate CXX)
#Would prefer to use CMAKE_CXX_STANDARD in CMake 3.1, but not yet
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
        message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

find_package(CUDA 6.0 QUIET)
if (${CUDA_FOUND})
  list(APPEND CUDA_NVCC_FLAGS "-arch=sm_30")
else (${CUDA_FOUND})
  message("CUDA not found on this system - building CPU-only")
endif (${CUDA_FOUND})

enable_testing()

add_subdirectory(source)

#add_subdirectory(test)
