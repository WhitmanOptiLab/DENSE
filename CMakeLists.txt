#Would prefer to make this 3.1, as it makes C++ dialect selection automatic.
cmake_minimum_required (VERSION 2.8)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Choose the type of build, options are: Debug Release
RelWithDebInfo MinSizeRel."
      FORCE)
endif(NOT CMAKE_BUILD_TYPE)

project (DDESimulatorTemplate CXX C)
#Would prefer to use CMAKE_CXX_STANDARD in CMake 3.1, but not yet
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wpedantic -Wc++11-compat -Wcomment -Waddress -Wchar-subscripts -Wformat -Winit-self -Wmaybe-uninitialized -Wnarrowing -Wnonnull -Wparentheses")
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
        message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

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
