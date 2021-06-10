if( NOT output_blessed )
   message( FATAL_ERROR "Variable output_blessed not defined" )
endif( NOT output_blessed )

if( NOT output_test )
   message( FATAL_ERROR "Variable output_test not defined" )
endif( NOT output_test )



separate_arguments( test_cmd )
execute_process(
   COMMAND ${test_cmd} RESULT_VARIABLE run_fail
)
if( run_fail )
   message( SEND_ERROR "Test simulation execution failure." )
endif( run_fail )

set(numdiff_call "../../test/numdiff-5.9.0/numdiff")
string(APPEND numdiff_call " -s ' \\\\t\\\\n,' -a 150")
separate_arguments( compare_command UNIX_COMMAND ${numdiff_call} )
execute_process(
   COMMAND ${compare_command} ${output_blessed} ${output_test}
   RESULT_VARIABLE test_not_successful
)

if( test_not_successful )
   message( SEND_ERROR "${output_test} does not match ${output_blessed} within plus or minus 100 of all numbers" )
endif( test_not_successful )
