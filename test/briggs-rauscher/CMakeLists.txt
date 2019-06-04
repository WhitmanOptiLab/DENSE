include_directories("${CMAKE_CURRENT_SOURCE_DIR}")

add_subdirectory("${PROJECT_SOURCE_DIR}/source" "${CMAKE_CURRENT_BINARY_DIR}/source")

SIMULATION(briggsrauscher_simulation ${CMAKE_CURRENT_SOURCE_DIR})
set(test_cmd ${CMAKE_CURRENT_BINARY_DIR}/briggs-rauscher_simulation)
set(test_args "--param-sets ${CMAKE_CURRENT_SOURCE_DIR}/param_set.csv --cell-total 100 --tissue-width 100 --time-total 10 --anlys-intvl 0.5 --step-size .01 --analysis ${CMAKE_CURRENT_SOURCE_DIR}/analyses.xml")
add_test( "briggs_rauscher_test"
 ${CMAKE_COMMAND}
 -D test_cmd=${test_cmd}
 -D test_args:string=${test_args}
 -D output_blessed=${CMAKE_CURRENT_SOURCE_DIR}/anlys_oscil.csv
 -D output_test=${CMAKE_CURRENT_BINARY_DIR}/anlys_oscil.csv
 -P ${CMAKE_SOURCE_DIR}/test/briggs_rauscher/run_test.cmake
 )

if (${CUDA_FOUND})
  CUDA_SIMULATION(briggsrauscher_cuda_simulation ${CMAKE_CURRENT_SOURCE_DIR})
endif (${CUDA_FOUND})

PARAM_SEARCH_LIB(briggsrauscher_param_search_lib ${CMAKE_CURRENT_SOURCE_DIR})

#add_executable(briggsrauscher_param_search search_main.cpp)

#target_link_libraries(briggsrauscher_param_search #briggsrauscher_param_search_lib briggsrauscher_simulation_lib)
