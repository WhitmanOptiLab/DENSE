include_directories("${CMAKE_CURRENT_SOURCE_DIR}") 

function(CSV_GEN localname simdir) 
  add_executable(${localname} csv_gen.cpp util/color.cpp io/csvw.cpp io/csvw_param.cpp)
  target_include_directories(${localname} PUBLIC ${simdir})
  target_include_directories(${localname} PUBLIC ${PROJECT_SOURCE_DIR}/source)
  target_include_directories(${localname} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
endfunction(CSV_GEN)

function(SIMULATION localname simdir)
  add_executable(${localname}
          ${PROJECT_SOURCE_DIR}/source/main.cpp
          ${PROJECT_SOURCE_DIR}/source/core/build_once.cpp
          ${PROJECT_SOURCE_DIR}/source/core/model.cpp
          ${PROJECT_SOURCE_DIR}/source/core/observable.cpp
          ${PROJECT_SOURCE_DIR}/source/anlys/basic.cpp
          ${PROJECT_SOURCE_DIR}/source/anlys/oscillation.cpp
          ${PROJECT_SOURCE_DIR}/source/sim/base.cpp
          ${PROJECT_SOURCE_DIR}/source/sim/cell_param.cpp
          ${PROJECT_SOURCE_DIR}/source/sim/determ/determ.cpp
          ${PROJECT_SOURCE_DIR}/source/sim/determ/baby_cl.cpp
          ${PROJECT_SOURCE_DIR}/source/sim/stoch/stoch.cpp
          ${PROJECT_SOURCE_DIR}/source/util/color.cpp
          ${PROJECT_SOURCE_DIR}/source/io/arg_parse.cpp
          ${PROJECT_SOURCE_DIR}/source/io/csvr.cpp
          ${PROJECT_SOURCE_DIR}/source/io/csvr_param.cpp
          ${PROJECT_SOURCE_DIR}/source/io/csvr_sim.cpp
          ${PROJECT_SOURCE_DIR}/source/io/csvw.cpp
          ${PROJECT_SOURCE_DIR}/source/io/csvw_param.cpp
          ${PROJECT_SOURCE_DIR}/source/io/csvw_sim.cpp)
  target_include_directories(${localname} PUBLIC ${simdir})
  target_include_directories(${localname} PUBLIC ${PROJECT_SOURCE_DIR}/source)
endfunction(SIMULATION localname simdir)

function(CUDA_SIMULATION localname simdir)
  CUDA_INCLUDE_DIRECTORIES(${simdir} ${PROJECT_SOURCE_DIR}/source)
  cuda_add_executable(${localname}
          ${PROJECT_SOURCE_DIR}/source/main_cuda.cu
          ${PROJECT_SOURCE_DIR}/source/core/build_once.cu
          ${PROJECT_SOURCE_DIR}/source/core/model.cpp
          ${PROJECT_SOURCE_DIR}/source/core/observable.cpp
          ${PROJECT_SOURCE_DIR}/source/sim/set_cuda.cu
          ${PROJECT_SOURCE_DIR}/source/sim/base.cpp
          ${PROJECT_SOURCE_DIR}/source/sim/determ/determ.cpp
          ${PROJECT_SOURCE_DIR}/source/sim/determ/determ_cuda.cu
          ${PROJECT_SOURCE_DIR}/source/sim/cell_param.cpp
          ${PROJECT_SOURCE_DIR}/source/sim/determ/baby_cl.cpp
          ${PROJECT_SOURCE_DIR}/source/sim/determ/baby_cl_cuda.cu
          ${PROJECT_SOURCE_DIR}/source/io/arg_parse.cpp
          ${PROJECT_SOURCE_DIR}/source/util/color.cpp
          ${PROJECT_SOURCE_DIR}/source/io/csvr.cpp
          ${PROJECT_SOURCE_DIR}/source/io/csvr_param.cpp
          ${PROJECT_SOURCE_DIR}/source/io/csvr_sim.cpp
          ${PROJECT_SOURCE_DIR}/source/io/csvw.cpp
          ${PROJECT_SOURCE_DIR}/source/io/csvw_param.cpp
          ${PROJECT_SOURCE_DIR}/source/io/csvw_sim.cpp)
  target_include_directories(${localname} PUBLIC ${simdir})
  target_include_directories(${localname} PUBLIC ${PROJECT_SOURCE_DIR}/source)
  target_link_libraries(${localname} ${CUDA_LIBRARIES})
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







