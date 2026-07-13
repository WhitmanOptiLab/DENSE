separate_arguments( test_args )

execute_process(
   COMMAND ${test_cmd} ${test_args} 
   RESULT_VARIABLE run_fail
)

if( run_fail )
   message( SEND_ERROR "Simulation failed to execute target binary." )
endif()