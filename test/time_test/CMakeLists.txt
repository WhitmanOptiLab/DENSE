##
# COMPARE TWO VERSIONS OF THE DENSE SIMULATOR
# Needs: 
#   - a compiled reference simulation from a previous version of DENSES titled "ref_sim" in the time_test dir
#   - a compiled test simulation from the current DENSE build titled "simulation" in the time_test dir
# To run: just run ctest from the current model build directory
##

include_directories("${CMAKE_CURRENT_SOURCE_DIR}")

#add_subdirectory("${PROJECT_SOURCE_DIR}/source" "${CMAKE_CURRENT_BINARY_DIR}/source")

SIMULATION(time_test ${CMAKE_CURRENT_SOURCE_DIR})

#the current DENSE build's compiled simulation called "simulation" 
set(simA_cmd ${CMAKE_SOURCE_DIR}/test/time_test/simulation)

#the previous DENSE build's compiled simulation called "ref_sim"
set(simB_cmd ${CMAKE_SOURCE_DIR}/test/time_test/ref_sim)

#the desired parameters for the test from the param_sets.csv file in this directory, be sure to update it with desired params
set(test_args "--param-sets ${CMAKE_SOURCE_DIR}/test/time_test/param_sets.csv --cell-total 10 --tissue-width 10 --time-total 30000 --anlys-intvl .01 --step-size .01")

add_test( "current_build_time"
 ${CMAKE_COMMAND}
 -D test_cmd=${simA_cmd}
 -D test_args:string=${test_args}
 -P ${CMAKE_SOURCE_DIR}/test/time_test/run_test.cmake
 )

add_test( "previous_build_time"
 ${CMAKE_COMMAND}
 -D test_cmd=${simB_cmd}
 -D test_args:string=${test_args}
 -P ${CMAKE_SOURCE_DIR}/test/time_test/run_test.cmake
 )
