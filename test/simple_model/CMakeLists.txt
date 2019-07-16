add_subdirectory(ndiff)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}")

add_subdirectory("${DENSE_SOURCE_DIR}/source" "${CMAKE_CURRENT_BINARY_DIR}/source")

GILLESPIE_SIMULATION(simple_gillespie ${CMAKE_CURRENT_SOURCE_DIR})
STOCH_NR_SIMULATION(simple_nr ${CMAKE_CURRENT_SOURCE_DIR})

add_test(NAME run-gillespie
        COMMAND ${CMAKE_CURRENT_BINARY_DIR}/simple_gillespie -p ${CMAKE_CURRENT_SOURCE_DIR}/param_sets.csv -t 100 -c 100 -w 10 -u 0.01 -d ${CMAKE_CURRENT_SOURCE_DIR}/init_conc.csv -a ${CMAKE_CURRENT_SOURCE_DIR}/analesysa.xml 
    
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

add_test(NAME run-next-reaction
        COMMAND ${CMAKE_CURRENT_BINARY_DIR}/simple_nr -p ${CMAKE_CURRENT_SOURCE_DIR}/param_sets.csv -t 100 -c 100 -w 10 -u 0.01 -d ${CMAKE_CURRENT_SOURCE_DIR}/init_conc.csv -a ${CMAKE_CURRENT_SOURCE_DIR}/analesysb.xml 
    
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

add_test(NAME gillespie-result
        COMMAND ${CMAKE_CURRENT_BINARY_DIR}/ndiff/maddiff gillespie.out test.ref test.cfg
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

add_test(NAME next-reaction-result
        COMMAND ${CMAKE_CURRENT_BINARY_DIR}/ndiff/maddiff nextreaction.out test.ref test.cfg
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)


set_tests_properties( gillespie-result
    PROPERTIES FAIL_REGULAR_EXPRESSION "diffs have been detected"
)

set_tests_properties( next-reaction-result
    PROPERTIES FAIL_REGULAR_EXPRESSION "diffs have been detected"
)
