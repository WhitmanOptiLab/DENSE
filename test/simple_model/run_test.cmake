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

if(${diffout} MATCHES "diffs have been detected")
   set (test_not_successful TRUE)
else()
   set (test_not_successful FALSE)
endif()

if( test_not_successful )
   message( SEND_ERROR "test.out does not match test.ref!" )
endif( test_not_successful )
