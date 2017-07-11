include_directories("${CMAKE_CURRENT_SOURCE_DIR}") 

function(CSV_GEN localname simdir) 
  add_executable(${localname} csv_gen.cpp util/color.cpp io/csvw.cpp io/csvw_param.cpp)
  target_include_directories(${localname} PUBLIC ${simdir})
endfunction(CSV_GEN)

function(SIMULATION localname simdir)
  add_executable(${localname}
          main.cpp
          sim/base.cpp
          sim/determ/determ.cpp
          sim/stoch/stoch.cpp
          ${simdir}/reaction.cpp
          sim/cell_param.cpp
          sim/determ/baby_cl.cpp
          io/arg_parse.cpp
          core/model.cpp
          util/color.cpp
          core/observable.cpp
          anlys/basic.cpp
          anlys/oscillation.cpp
          io/csvr.cpp
          io/csvr_param.cpp
          io/csvr_sim.cpp
          io/csvw.cpp
          io/csvw_param.cpp
          io/csvw_sim.cpp)
  target_include_directories(${localname} PUBLIC ${simdir})
  target_include_directories(${localname} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
endfunction(SIMULATION localname simdir)

function(CUDA_SIMULATION localname simdir)
  cuda_add_executable(${localname}
          test_simulation_set_cuda.cu
          simulation_set_cuda.cu
          simulation_base.cpp
          simulation_determ.cpp
          simulation_cuda.cu
          ${simdir}/reaction.cu
          cell_param.cpp
          baby_cl.cpp
          baby_cl_cuda.cu
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
endfunction(CUDA_SIMULATION localname simdir)

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
    SIMULATION(simulation ${PROJECT_BINARY_DIR})
    if (${CUDA_FOUND})
      CUDA_SIMULATION(cuda_simulation ${PROJECT_BINARY_DIR})
    endif (${CUDA_FOUND})
    # Forces csv_gen to run before all the test_'s are built.
    # This occurs when 'make' is run.
    add_custom_target(csv_gen_run ${PROJECT_BINARY_DIR}/source/csv_gen ../)
    add_dependencies(csv_gen_run csv_gen)
    add_dependencies(simulation csv_gen_run)
  endif (${BUILD_TESTS_ONLY})
#add_test(ctest_simulation main -p ../param_list.csv -c 200 -w 50 -s 0.01 -a 0.1 -r 10 -t 60 -e ../data_out.csv)
endif(${CMAKE_CURRENT_BINARY_DIR} STREQUAL ${PROJECT_BINARY_DIR}/source)







