#Would prefer to make this 3.1, as it makes C++ dialect selection automatic.
#changed to 3.9 to mirror readme requirements
cmake_minimum_required (VERSION 3.9)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Choose the type of build, options are: Debug Release
RelWithDebInfo MinSizeRel."
      FORCE)
endif(NOT CMAKE_BUILD_TYPE)

project (DDESimulatorTemplate CXX C) 
set(DENSE_SOURCE_DIR "${PROJECT_SOURCE_DIR}")
set(DENSE_SOURCE_DIR "${PROJECT_SOURCE_DIR}" PARENT_SCOPE)
#Would prefer to use CMAKE_CXX_STANDARD in CMake 3.1, but not yet
include(CheckCXXCompilerFlag)
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.11")
  set(CMAKE_CXX_STANDARD 20 PARENT_SCOPE)
else()
  set(CMAKE_CXX_STANDARD 14 PARENT_SCOPE)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED 14 PARENT_SCOPE)
if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ferror-limit=3 -Wlogical-op-parentheses -Wnoexcept-type")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fmax-errors=3 -Wlogical-op -Wnoexcept -Wstrict-null-sentinel -Wuseless-cast -Wsuggest-override")
endif()
string(REPLACE ";" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} "
 "-g -pedantic -Wall -Wextra -Werror -Wno-cast-align -Wcast-qual -Wctor-dtor-privacy "
 "-Wdisabled-optimization -Wformat=2 -Winit-self -Wmissing-include-dirs -Wno-type-limits "
 "-Woverloaded-virtual -Wredundant-decls -Wstrict-overflow=1 "
 "-Wswitch-default -Wundef -Wunused -DMPI=1")

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



find_package(MPI QUIET)
if (${MPI_FOUND})
  message(STATUS "Run: ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} EXECUTABLE ${MPIEXEC_POSTFLAGS} ARGS ${MPI_CXX_COMPILE_FLAGS}")
else (${MPI_FOUND})
  message("MPI not found")
endif (${MPI_FOUND})

set(BUILD_TESTS_ONLY FALSE)
if (NOT EXISTS ${PROJECT_BINARY_DIR}/reaction_deltas.hpp)
  message("No reaction_deltas.cpp found in build directory")
  set(BUILD_TESTS_ONLY TRUE)
endif (NOT EXISTS ${PROJECT_BINARY_DIR}/reaction_deltas.hpp)
if (NOT EXISTS ${PROJECT_BINARY_DIR}/model_impl.hpp)
  message("No model_impl.hpp found in build directory")
  set(BUILD_TESTS_ONLY TRUE)
endif (NOT EXISTS ${PROJECT_BINARY_DIR}/model_impl.hpp)
if (NOT EXISTS ${PROJECT_BINARY_DIR}/reactions_list.hpp)
  message("No reactions_list.hpp found in build directory")
  set(BUILD_TESTS_ONLY TRUE)
endif (NOT EXISTS ${PROJECT_BINARY_DIR}/reactions_list.hpp)
if (NOT EXISTS ${PROJECT_BINARY_DIR}/specie_list.hpp)
  message("No specie_list.hpp found in build directory")
  set(BUILD_TESTS_ONLY TRUE)
endif (NOT EXISTS ${PROJECT_BINARY_DIR}/specie_list.hpp)

option(BUILD_ALL_METHODS "Build executibles for all simulation methods" ON)
option(DETERMINISTIC_SIM "" ON)
option(GILLESPIE_SIM "" ${BUILD_ALL_METHODS})
option(SLOW_GILLESPIE_SIM "" ${BUILD_ALL_METHODS})
option(NEXTREACTION_SIM "" ON)
option(SIM_SORTING_DIRECT_HPP "" ON)
option(SORTING_DIRECT_SIM "" ON)
option(SIM_LOG_DIRECT_HPP "" ON)
option(REJECTION_BASED_SIMULATION_HPP "" ON)
option(CUDA_SIM "" ${BUILD_ALL_METHODS})
option(MPI_PARAM_SEARCH "" ${BUILD_ALL_METHODS})
option(PARAM_SEARCH "" ${BUILD_ALL_METHODS})
option(IN_MEM_LOG "" ON)

enable_testing()

add_subdirectory(source)

if (${BUILD_TESTS_ONLY})
  add_subdirectory(test)
endif (${BUILD_TESTS_ONLY})

