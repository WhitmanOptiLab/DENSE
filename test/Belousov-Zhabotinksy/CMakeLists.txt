include_directories("${CMAKE_CURRENT_SOURCE_DIR}")

add_subdirectory("${DENSE_SOURCE_DIR}/source" "${CMAKE_CURRENT_BINARY_DIR}/source")

DETERMINISTIC_SIMULATION(bz_simulation ${CMAKE_CURRENT_SOURCE_DIR})
set(test_cmd ${CMAKE_CURRENT_BINARY_DIR}/bz_simulation)
set(test_args "--param-sets ${CMAKE_CURRENT_SOURCE_DIR}/param_sets.csv --cell-total 100 --tissue-width 100 --time-total 15 --anlys-intvl 0.5 -r 279507039 --analysis ${CMAKE_CURRENT_SOURCE_DIR}/analesys.xml")
add_test( "oscil_test_pt1"
 ${CMAKE_COMMAND}
 -D test_cmd=${test_cmd}
 -D test_args:string=${test_args}
 -D output_blessed=${CMAKE_CURRENT_SOURCE_DIR}/anlys_basic_a.csv
 -D output_test=${CMAKE_CURRENT_BINARY_DIR}/anlys_basic_a.csv
 -P ${CMAKE_SOURCE_DIR}/test/her_model_2014/run_test.cmake
 )
add_test( "oscil_test_pt2"
 ${CMAKE_COMMAND}
 -D test_cmd=${test_cmd}
 -D test_args:string=${test_args}
 -D output_blessed=${CMAKE_CURRENT_SOURCE_DIR}/anlys_basic_b.csv
 -D output_test=${CMAKE_CURRENT_BINARY_DIR}/anlys_basic_b.csv
 -P ${CMAKE_SOURCE_DIR}/test/her_model_2014/run_test.cmake
 )
add_test( "oscil_test_pt3"
 ${CMAKE_COMMAND}
 -D test_cmd=${test_cmd}
 -D test_args:string=${test_args}
 -D output_blessed=${CMAKE_CURRENT_SOURCE_DIR}/anlys_basic_c.csv
 -D output_test=${CMAKE_CURRENT_BINARY_DIR}/anlys_basic_c.csv
 -P ${CMAKE_SOURCE_DIR}/test/her_model_2014/run_test.cmake
 )
add_test( "oscil_test_pt4"
 ${CMAKE_COMMAND}
 -D test_cmd=${test_cmd}
 -D test_args:string=${test_args}
 -D output_blessed=${CMAKE_CURRENT_SOURCE_DIR}/anlys_basic_d.csv
 -D output_test=${CMAKE_CURRENT_BINARY_DIR}/anlys_basic_d.csv
 -P ${CMAKE_SOURCE_DIR}/test/her_model_2014/run_test.cmake
 )
add_test( "oscil_test_pt5"
 ${CMAKE_COMMAND}
 -D test_cmd=${test_cmd}
 -D test_args:string=${test_args}
 -D output_blessed=${CMAKE_CURRENT_SOURCE_DIR}/anlys_basic_e.csv
 -D output_test=${CMAKE_CURRENT_BINARY_DIR}/anlys_basic_e.csv
 -P ${CMAKE_SOURCE_DIR}/test/her_model_2014/run_test.cmake
 )
add_test( "oscil_test_overall"
 ${CMAKE_COMMAND}
 -D test_cmd=${test_cmd}
 -D test_args:string=${test_args}
 -D output_blessed=${CMAKE_CURRENT_SOURCE_DIR}/anlys_oscil.csv
 -D output_test=${CMAKE_CURRENT_BINARY_DIR}/anlys_oscil.csv
 -P ${CMAKE_SOURCE_DIR}/test/her_model_2014/run_test.cmake
 )
if (${CUDA_FOUND})
  CUDA_SIMULATION(turing_cuda_simulation ${CMAKE_CURRENT_SOURCE_DIR})
endif (${CUDA_FOUND})

#PARAM_SEARCH_LIB(turing_param_search_lib ${CMAKE_CURRENT_SOURCE_DIR})

#add_executable(turing_param_search search_main.cpp)

#target_link_libraries(turing_param_search turing_param_search_lib turing_simulation_lib)
