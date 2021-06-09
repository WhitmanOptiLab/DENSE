include_directories("${CMAKE_CURRENT_SOURCE_DIR}")

add_subdirectory("${DENSE_SOURCE_DIR}/source" "${CMAKE_CURRENT_BINARY_DIR}/source")

BUILTIN_SIM_CLI(nan_simulation ${CMAKE_CURRENT_SOURCE_DIR}
"Deterministic_Simulation")
set(test_cmd ${CMAKE_CURRENT_BINARY_DIR}/nan_simulation)
set(test_args "--param-sets ${CMAKE_CURRENT_SOURCE_DIR}/param_sets.csv --cell-total 100 --tissue-width 100 --time-total 15 --anlys-intvl 0.5 -s 0.01 --analysis ${CMAKE_CURRENT_SOURCE_DIR}/analyses.xml")
add_test( "nan_test"
 ${CMAKE_COMMAND}
 -D test_cmd=${test_cmd}
 -D test_args:string=${test_args}
 -D output_blessed=${CMAKE_CURRENT_SOURCE_DIR}/anlys_basic_a.csv
 -D output_test=${CMAKE_CURRENT_BINARY_DIR}/anlys_basic_a.csv
 -P ${CMAKE_SOURCE_DIR}/test/her_model_2014/run_test.cmake
 )
if (${CUDA_FOUND})
  CUDA_SIMULATION(nan_cuda_simulation ${CMAKE_CURRENT_SOURCE_DIR})
endif (${CUDA_FOUND})

#PARAM_SEARCH_LIB(nan_param_search_lib ${CMAKE_CURRENT_SOURCE_DIR})

#add_executable(nan_param_search search_main.cpp)

#target_link_libraries(nan_param_search nan_param_search_lib nan_simulation_lib)
