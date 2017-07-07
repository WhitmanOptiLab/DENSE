include_directories("${CMAKE_CURRENT_SOURCE_DIR}") 

function(CSV_GEN localname simdir) 
  add_executable(${localname} csv_gen.cpp color.cpp csvw.cpp csvw_param.cpp)
  target_include_directories(${localname} PUBLIC ${simdir})
endfunction(CSV_GEN)

function(TEST_SIM localname simdir) 
  add_executable(${localname}
          test_simulation.cpp
          simulation_base.cpp
          simulation_determ.cpp
          simulation_stoch.cpp
          ${simdir}/reaction.cpp
          cell_param.cpp
          baby_cl.cpp
          arg_parse.cpp
          model.cpp
          color.cpp
          observable.cpp
          analysis.cpp
          csvr.cpp
          csvr_param.cpp
          csvr_sim.cpp
          csvw.cpp
          csvw_param.cpp
          csvw_sim.cpp)
  target_include_directories(${localname} PUBLIC ${simdir})
endfunction(TEST_SIM)

function(TEST_SIM_SET localname simdir) 
  add_executable(${localname}
          test_simulation_set.cpp
          simulation_base.cpp
          simulation_determ.cpp
          simulation_stoch.cpp
          ${simdir}/reaction.cpp
          cell_param.cpp
          baby_cl.cpp
          arg_parse.cpp
          model.cpp
          color.cpp
          observable.cpp
          csvr.cpp
          csvr_param.cpp
          csvr_sim.cpp
          csvw.cpp
          csvw_param.cpp
          csvw_sim.cpp)
  target_include_directories(${localname} PUBLIC ${simdir})
endfunction(TEST_SIM_SET)

function(TEST_MAIN localname simdir)
  add_executable(${localname}
          test_main.cpp
          simulation_base.cpp
          simulation_determ.cpp
          simulation_stoch.cpp
          ${simdir}/reaction.cpp
          cell_param.cpp
          baby_cl.cpp
          arg_parse.cpp
          model.cpp
          color.cpp
          observable.cpp
          analysis.cpp
          csvr.cpp
          csvr_param.cpp
          csvr_sim.cpp
          csvw.cpp
          csvw_param.cpp
          csvw_sim.cpp)
  target_include_directories(${localname} PUBLIC ${simdir})
endfunction(TEST_MAIN localname simdir)

if( ${CMAKE_CURRENT_BINARY_DIR} STREQUAL ${PROJECT_BINARY_DIR}/source)
  set(BUILD_TESTS_ONLY FALSE)
  if (NOT EXISTS ${PROJECT_BINARY_DIR}/reaction.cpp) 
    message("No reaction.cpp found in build directory")
    set(BUILD_TESTS_ONLY TRUE)
  endif (NOT EXISTS ${PROJECT_BINARY_DIR}/reaction.cpp) 
  if (NOT EXISTS ${PROJECT_BINARY_DIR}/model_impl.hpp) 
    message("No model_impl.hpp found in build directory")
    set(BUILD_TESTS_ONLY TRUE)
  endif (NOT EXISTS ${PROJECT_BINARY_DIR}/model_impl.hpp) 
  if (NOT EXISTS ${PROJECT_BINARY_DIR}/reactions_list.hpp) 
    message("No reactions_list.hpp found in build directory")
    set(BUILD_TESTS_ONLY TRUE)
  endif (NOT EXISTS ${PROJECT_BINARY_DIR}/reactions_list.hpp) 
  if (NOT EXISTS ${PROJECT_BINARY_DIR}/specie_list.hpp) 
    message("No specie_list.hpp found in build directory")
    set(BUILD_TESTS_ONLY TRUE)
  endif (NOT EXISTS ${PROJECT_BINARY_DIR}/specie_list.hpp) 

  if (${BUILD_TESTS_ONLY})
    message(WARNING "No or incomplete model implementation found - building test cases only")
  else (${BUILD_TESTS_ONLY})
    CSV_GEN(csv_gen ${PROJECT_BINARY_DIR})
    TEST_SIM(test_simulation ${PROJECT_BINARY_DIR})
    TEST_SIM_SET(test_simulation_set ${PROJECT_BINARY_DIR})
    TEST_MAIN(test_main ${PROJECT_BINARY_DIR})
    # Forces csv_gen to run before all the test_'s are built.
    # This occurs when 'make' is run.
    add_custom_target(csv_gen_run ${PROJECT_BINARY_DIR}/source/csv_gen ../)
    add_dependencies(csv_gen_run csv_gen)

    add_dependencies(test_simulation csv_gen_run)
    # Warns user about lack of param_list.csv
    if(NOT EXISTS "param_list.csv")
      message(WARNING 
"Upon running 'make', if it doesn't already exist, 'param_list_template.csv' 
will be generated in the model directory. Please fill out this template then 
rename it to 'param_list.csv' before attempting to run any compiled tests or 
executables."
      )
    endif()
    #add_dependencies(test_simulation_set csv_gen_run)
    #add_dependencies(test_main csv_gen_run)
  endif (${BUILD_TESTS_ONLY})
#add_test(ctest_simulation test_simulation -p ../param_list.csv)
#add_test(ctest_simulation_set test_simulation_set -p ../param_list.csv)
#add_test(ctest_main test_main -p ../param_list.csv -c 200 -w 50 -s 0.01 -a 0.1 -r 10 -t 60 -e ../data_out.csv)
endif(${CMAKE_CURRENT_BINARY_DIR} STREQUAL ${PROJECT_BINARY_DIR}/source)







