include_directories("${CMAKE_CURRENT_SOURCE_DIR}")

add_subdirectory("${PROJECT_SOURCE_DIR}/source" "${CMAKE_CURRENT_BINARY_DIR}/source")

SIMULATION(her2014_simulation ${CMAKE_CURRENT_SOURCE_DIR})
if (${CUDA_FOUND})
  CUDA_SIMULATION(her2014_cuda_simulation ${CMAKE_CURRENT_SOURCE_DIR})
endif (${CUDA_FOUND})
