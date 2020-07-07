if( NOT output_blessed )
   message( FATAL_ERROR "Variable output_blessed not defined" )
endif( NOT output_blessed )

if( NOT output_test )
   message( FATAL_ERROR "Variable output_test not defined" )
endif( NOT output_test )

set(numdiff_call ${PROJECT_BINARY_DIR}/test/numdiff-5.9.0/numdiff)
set(numdiff_args "-s \" \\t\\n,\" -a 1")

message(${numdiff_call} ${numdiff_args} )
separate_arguments( numdiff_args )

execute_process(
   COMMAND ${numdiff_call} ${numdiff_args} ${output_blessed} ${output_test}
   RESULT_VARIABLE test_not_successful
)

if( test_not_successful )
   message( SEND_ERROR "${output_test} does not match ${output_blessed} within plus or minus 1 of all numbers" )
endif( test_not_successful )