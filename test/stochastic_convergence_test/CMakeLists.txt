include_directories("${CMAKE_CURRENT_SOURCE_DIR}")

add_subdirectory("${DENSE_SOURCE_DIR}/source" "${CMAKE_CURRENT_BINARY_DIR}/source")

BUILTIN_SIM_CLI(stochastic_convergence_simulation ${CMAKE_CURRENT_SOURCE_DIR} "Fast_Gillespie_Direct_Simulation")

set(test_cmd "${CMAKE_CURRENT_BINARY_DIR}/stochastic_convergence_simulation")
string(APPEND test_cmd " --param-sets ${CMAKE_CURRENT_SOURCE_DIR}/param_sets.csv --cell-total 100 --tissue-width 100 --time-total 10 --anlys-intvl 0.1 --analysis ${CMAKE_CURRENT_SOURCE_DIR}/analyses.xml")
add_test( "stochastic_convergence_test"
 ${CMAKE_COMMAND}
 -D test_cmd=${test_cmd}
 -D output_blessed=${CMAKE_CURRENT_SOURCE_DIR}/anlys_converge.csv
 -D output_test=${CMAKE_CURRENT_BINARY_DIR}/anlys_converge.csv
 -P ${CMAKE_SOURCE_DIR}/test/stochastic_convergence_test/run_test.cmake
 )
