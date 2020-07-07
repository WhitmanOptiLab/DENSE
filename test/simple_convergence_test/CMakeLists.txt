include_directories("${CMAKE_CURRENT_SOURCE_DIR}")

add_subdirectory("${DENSE_SOURCE_DIR}/source" "${CMAKE_CURRENT_BINARY_DIR}/source")

DETERMINISTIC_SIMULATION(convergence_simulation ${CMAKE_CURRENT_SOURCE_DIR})

set(test_cmd ${CMAKE_CURRENT_BINARY_DIR}/convergence_simulation)
set(test_args "--param-sets ${CMAKE_CURRENT_SOURCE_DIR}/param_sets.csv --cell-total 100 --tissue-width 100 --time-total 10 --anlys-intvl 0.01 --step-size .01 --analysis ${CMAKE_CURRENT_SOURCE_DIR}/analyses.xml")
add_test( "basic_convergence_test"
 ${CMAKE_COMMAND}
 -D test_cmd=${test_cmd}
 -D test_args:string=${test_args}
 -D output_blessed=${CMAKE_CURRENT_SOURCE_DIR}/anlys_converge.csv
 -D output_test=${CMAKE_CURRENT_BINARY_DIR}/anlys_converge.csv
 -P ${CMAKE_SOURCE_DIR}/test/her_model_2014/run_test.cmake
 )
if (${CUDA_FOUND})
  CUDA_SIMULATION(turing_cuda_simulation ${CMAKE_CURRENT_SOURCE_DIR})
endif (${CUDA_FOUND})
