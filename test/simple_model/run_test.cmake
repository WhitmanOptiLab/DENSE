message(${test_cmd} ${test_args} )
separate_arguments( test_args )

execute_process(
   COMMAND ${test_cmd} ${test_args} 
   OUTPUT_VARIABLE run 
   RESULT_VARIABLE run_fail
)

message (${cmp} ${args})
separate_arguments(args)

execute_process(
   COMMAND ${cmp} ${args}
   ERROR_VARIABLE diffout
)

if(run_fail)
   message( SEND_ERROR "error running simulation" )
endif()

if(${diffout} MATCHES "diffs have been detected")
   message( SEND_ERROR "test.out does not match test.ref!" )
endif()

