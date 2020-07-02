message(${test_cmd} ${test_args} )
separate_arguments( test_args )

execute_process(
   COMMAND ${test_cmd} ${test_args} RESULT_VARIABLE run_fail
)

if( run_fail )
   message( SEND_ERROR "Error in Simulation, check that directory paths are correct" )
endif( test_not_successful )
