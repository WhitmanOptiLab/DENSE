include_directories("${CMAKE_CURRENT_SOURCE_DIR}")

add_subdirectory("${DENSE_SOURCE_DIR}/source" "${CMAKE_CURRENT_BINARY_DIR}/source")

function(STOCHASTIC_CONVERGENCE_TEST method name)
  BUILTIN_SIM_CLI(${name}_convergence_simulation ${CMAKE_CURRENT_SOURCE_DIR} ${method})

  set(test_cmd "${CMAKE_CURRENT_BINARY_DIR}/${name}_convergence_simulation")
  string(APPEND test_cmd " --param-sets ${CMAKE_CURRENT_SOURCE_DIR}/param_sets.csv --cell-total 100 --tissue-width 100 --time-total 10 --anlys-intvl 0.1 --analysis ${CMAKE_CURRENT_SOURCE_DIR}/analyses.xml")
  add_test( "${name}_convergence_test"
      ${CMAKE_COMMAND}
      -D test_cmd=${test_cmd}
      -D output_blessed=${CMAKE_CURRENT_SOURCE_DIR}/anlys_converge.csv
      -D output_test=${CMAKE_CURRENT_BINARY_DIR}/anlys_converge.csv
      -P ${CMAKE_SOURCE_DIR}/test/stochastic_convergence_test/run_test.cmake
      )
endfunction(STOCHASTIC_CONVERGENCE_TEST method name)

#STOCHASTIC_CONVERGENCE_TEST("Stochastic_Simulation" "dir")
STOCHASTIC_CONVERGENCE_TEST("Fast_Gillespie_Direct_Simulation" "fastdir")
STOCHASTIC_CONVERGENCE_TEST("Next_Reaction_Simulation" "nr")
STOCHASTIC_CONVERGENCE_TEST("Sorting_Direct_Simulation" "sorting_dir_")
STOCHASTIC_CONVERGENCE_TEST("Rejection_Based_Simulation" "RSSA_")
STOCHASTIC_CONVERGENCE_TEST("Log_Direct_Simulation" "log_dir_")
