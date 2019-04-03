include_directories("${CMAKE_CURRENT_SOURCE_DIR}")

add_subdirectory("${PROJECT_SOURCE_DIR}/source" "${CMAKE_CURRENT_BINARY_DIR}/source")

SIMULATION(her2014_simulation ${CMAKE_CURRENT_SOURCE_DIR})
if (${CUDA_FOUND})
  CUDA_SIMULATION(her2014_cuda_simulation ${CMAKE_CURRENT_SOURCE_DIR})
endif (${CUDA_FOUND})

PARAM_SEARCH_LIB(her2014_param_search_lib ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(her2014_param_search search_main.cpp)

target_link_libraries(her2014_param_search her2014_param_search_lib her2014_simulation_lib)

enable_testing()

add_test(her_test her2014_param_search)
set_test_properties(her_test PROPERTIES PASS_REGULAR_EXPRESSION "")
