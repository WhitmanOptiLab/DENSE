include_directories("${CMAKE_CURRENT_SOURCE_DIR}")

add_subdirectory("${PROJECT_SOURCE_DIR}/source" "${CMAKE_CURRENT_BINARY_DIR}/source")

SIMULATION(her2014_simulation ${CMAKE_CURRENT_SOURCE_DIR})
add_test( "her_test"
 ${CMAKE_COMMAND}
 -Dtest_cmd=${CMAKE_CURRENT_BINARY_DIR}/her2014_simulation|--param-sets|${CMAKE_CURRENT_SOURCE_DIR}/param_sets.csv|--cell-total|10|--tissue-width|5|--time-total|3000|--anlys-intvl|.01|--step-size|.01|--analysis|${CMAKE_CURRENT_SOURCE_DIR}/analyses.xml
 -Doutput_blessed=${CMAKE_CURRENT_SOURCE_DIR}/anlys_basic_a.csv
 -Doutput_test=${CMAKE_CURRENT_BINARY_DIR}/anlys_basic_a.csv
 -P ${CMAKE_SOURCE_DIR}/test/her_model_2014/run_test.cmake
 )

if (${CUDA_FOUND})
  CUDA_SIMULATION(her2014_cuda_simulation ${CMAKE_CURRENT_SOURCE_DIR})
endif (${CUDA_FOUND})

PARAM_SEARCH_LIB(her2014_param_search_lib ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(her2014_param_search search_main.cpp)

target_link_libraries(her2014_param_search her2014_param_search_lib her2014_simulation_lib)
