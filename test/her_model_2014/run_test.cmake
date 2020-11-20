if( NOT output_blessed )
   message( FATAL_ERROR "Variable output_blessed not defined" )
endif( NOT output_blessed )

if( NOT output_test )
   message( FATAL_ERROR "Variable output_test not defined" )
endif( NOT output_test )

if( NOT tolerance )
  set(tolerance "0")
endif( NOT tolerance )
message(${test_cmd} ${test_args} )
separate_arguments( test_args )


execute_process(
   COMMAND ${test_cmd} ${test_args} RESULT_VARIABLE run_fail
)

message( "Tolerance is" ${tolerance} )

execute_process (
   COMMAND ${CMAKE_CURRENT_BINARY_DIR}/../numdiff-5.9.0/numdiff -s ,\n -r ${tolerance} ${output_blessed} ${output_test}
   RESULT_VARIABLE test_not_successful
)

if( test_not_successful )
   message( SEND_ERROR "${output_test} does not match ${output_blessed}!" )
endif( test_not_successful )
