cmake_minimum_required (VERSION 3.9)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Choose the type of build, options are: Debug Release
RelWithDebInfo MinSizeRel."
      FORCE)
endif(NOT CMAKE_BUILD_TYPE)

project (DDESimulatorTemplate CMAKE_CXX_STANDARD LANGUAGES CXX;C)
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
string(REPLACE ";" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} "
 "-g -fconcepts -pedantic -fmax-errors=3 -Wall -Wextra -Wcast-align -Wcast-qual -Wctor-dtor-privacy "
 "-Wdisabled-optimization -Wformat=2 -Winit-self -Wlogical-op -Wmissing-include-dirs "
 "-Wnoexcept -Woverloaded-virtual -Wredundant-decls -Wstrict-null-sentinel -Wstrict-overflow=5 "
 "-Wswitch-default -Wundef -Wunused -Wzero-as-null-pointer-constant -Wuseless-cast -Wsuggest-override -fopenmp")
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
        message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

#Checking if compiler has OpenMP Support
find_package(OpenMP REQUIRED)
add_executable(DDESimulatorTemplate DDESimulatorTemplate.cc)
target_link_libraries(DDESimulatorTemplate PRIVATE OpenMP::OpenMP_CXX)

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
