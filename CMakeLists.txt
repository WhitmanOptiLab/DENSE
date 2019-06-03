cmake_minimum_required (VERSION 3.9)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Choose the type of build, options are: Debug Release
RelWithDebInfo MinSizeRel."
      FORCE)
endif(NOT CMAKE_BUILD_TYPE)

project (DDESimulatorTemplate LANGUAGES CXX;C)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

string(REPLACE ";" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} "
 "-g -fconcepts -pedantic -fmax-errors=3 -Wall -Wextra -Wcast-align -Wcast-qual -Wctor-dtor-privacy "
 "-Wdisabled-optimization -Wformat=2 -Winit-self -Wlogical-op -Wmissing-include-dirs "
 "-Wnoexcept -Woverloaded-virtual -Wredundant-decls -Wstrict-null-sentinel -Wstrict-overflow=5 "
 "-Wswitch-default -Wundef -Wunused -Wzero-as-null-pointer-constant -Wuseless-cast -Wsuggest-override")

#Checking if compiler has OpenMP Support
find_package(OpenMP REQUIRED)

set(Real "double" CACHE STRING "dense::Real")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDENSE_CONFIGURABLE_Real=\"${Real}\"")

find_package(CUDA 6.0 QUIET)
if (${CUDA_FOUND})
  SET(CUDA_PROPAGATE_HOST_FLAGS OFF)
  set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
  set(CUDA_NVCC_FLAGS "-g;-arch=sm_30;--relocatable-device-code=true" )
  if (CMAKE_VERSION VERSION_LESS 3.0.2)
    list(APPEND CUDA_NVCC_FLAGS "-std=c++11")
  endif (CMAKE_VERSION VERSION_LESS 3.0.2)
  set(CUDA_SEPARABLE_COMPILATION ON)
else (${CUDA_FOUND})
  message("CUDA not found on this system - building CPU-only")
endif (${CUDA_FOUND})

enable_testing()

add_subdirectory(source)

add_subdirectory(test)
