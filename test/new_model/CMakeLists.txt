add_subdirectory("${PROJECT_SOURCE_DIR}/source" "${CMAKE_CURRENT_BINARY_DIR}/source")

SIMULATION(newmodel_simulation ${CMAKE_CURRENT_SOURCE_DIR})
if (${CUDA_FOUND})
  CUDA_SIMULATION(newmodel_cuda_simulation ${CMAKE_CURRENT_SOURCE_DIR})
endif (${CUDA_FOUND})
