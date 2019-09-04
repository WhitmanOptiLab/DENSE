include_directories("${CMAKE_CURRENT_SOURCE_DIR}")

add_subdirectory("${DENSE_SOURCE_DIR}/source" "${CMAKE_CURRENT_BINARY_DIR}/source")

GILLESPIE_SIMULATION(simple_gillespie ${CMAKE_CURRENT_SOURCE_DIR})
STOCH_NR_SIMULATION(simple_nr ${CMAKE_CURRENT_SOURCE_DIR})
REJECTION_BASED_SIMULATION(simple_rej ${CMAKE_CURRENT_SOURCE_DIR})
LOG_DIRECT_SIMULATION(simple_log ${CMAKE_CURRENT_SOURCE_DIR})

set (test_args "-p ${CMAKE_CURRENT_SOURCE_DIR}/param_sets.csv -t 100 -c 100 -w 10 -u 0.01 -d ${CMAKE_CURRENT_SOURCE_DIR}/init_conc.csv -a ${CMAKE_CURRENT_SOURCE_DIR}/analesys.xml")
set (ndiff_cmd "${CMAKE_CURRENT_SOURCE_DIR}/maddiff ")
set (ndiff_args "${CMAKE_CURRENT_BINARY_DIR}/test.out ${CMAKE_CURRENT_SOURCE_DIR}/test.ref ${CMAKE_CURRENT_SOURCE_DIR}/test.cfg")

set (run_gillespie_cmd ${CMAKE_CURRENT_BINARY_DIR}/simple_gillespie)

add_test("simple-gillespie-test"
 ${CMAKE_COMMAND} 
 -D test_cmd=${run_gillespie_cmd} 
 -D test_args:string=${test_args}
 -D cmp=${ndiff_cmd}
 -D args:string=${ndiff_args}
 -P ${CMAKE_SOURCE_DIR}/test/simple_model/run_test.cmake
)

set (run_nr_cmd ${CMAKE_CURRENT_BINARY_DIR}/simple_nr)

add_test("simple-nr-test"
 ${CMAKE_COMMAND} 
 -D test_cmd=${run_nr_cmd} 
 -D test_args:string=${test_args}
 -D cmp=${ndiff_cmd}
 -D args:string=${ndiff_args}
 -P ${CMAKE_SOURCE_DIR}/test/simple_model/run_test.cmake
)

set (run_rej_cmd ${CMAKE_CURRENT_BINARY_DIR}/simple_rej)

add_test("simple-rej-test"
 ${CMAKE_COMMAND} 
 -D test_cmd=${run_rej_cmd} 
 -D test_args:string=${test_args}
 -D cmp=${ndiff_cmd}
 -D args:string=${ndiff_args}
 -P ${CMAKE_SOURCE_DIR}/test/simple_model/run_test.cmake
)

set (run_log_cmd ${CMAKE_CURRENT_BINARY_DIR}/simple_log)

add_test("log-direct-test"
 ${CMAKE_COMMAND} 
 -D test_cmd=${run_log_cmd} 
 -D test_args:string=${test_args}
 -D cmp=${ndiff_cmd}
 -D args:string=${ndiff_args}
 -P ${CMAKE_SOURCE_DIR}/test/simple_model/run_test.cmake
)
