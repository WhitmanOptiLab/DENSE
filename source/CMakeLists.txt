include_directories("${CMAKE_CURRENT_SOURCE_DIR}")

function(CSV_GEN localname simdir)
  add_executable(${localname} csv_gen.cpp utility/color.cpp io/csvw.cpp)
  target_include_directories(${localname} PUBLIC ${simdir})
  target_include_directories(${localname} PUBLIC ${PROJECT_SOURCE_DIR}/source)
  target_include_directories(${localname} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
endfunction(CSV_GEN)

function(SIMCORELIB localname simdir)
  if (NOT TARGET ${localname})
    add_library(${localname}
          ${PROJECT_SOURCE_DIR}/source/core/build_once.cpp
          ${PROJECT_SOURCE_DIR}/source/sim/base.cpp
          ${PROJECT_SOURCE_DIR}/source/sim/cell_param.cpp
          ${PROJECT_SOURCE_DIR}/source/sim/determ/determ.cpp
          ${PROJECT_SOURCE_DIR}/source/sim/determ/baby_cl.cpp
          ${PROJECT_SOURCE_DIR}/source/sim/stoch/stoch.cpp
          ${PROJECT_SOURCE_DIR}/source/utility/common_utils.cpp
          ${PROJECT_SOURCE_DIR}/source/utility/color.cpp
          ${PROJECT_SOURCE_DIR}/source/utility/preprocessor.cpp
          ${PROJECT_SOURCE_DIR}/source/io/arg_parse.cpp
          ${PROJECT_SOURCE_DIR}/source/io/csvr.cpp
          ${PROJECT_SOURCE_DIR}/source/io/csvr_sim.cpp
          ${PROJECT_SOURCE_DIR}/source/io/csvw.cpp
          ${PROJECT_SOURCE_DIR}/source/io/ezxml/ezxml.c)

    target_include_directories(${localname} PUBLIC ${simdir})
    target_include_directories(${localname} PUBLIC ${PROJECT_SOURCE_DIR}/source)
  endif (NOT TARGET ${localname})
endfunction(SIMCORELIB localname simdir)

function(SIMULATION localname simdir)
  SIMCORELIB(${localname}_lib ${simdir})
  add_executable(${localname} ${PROJECT_SOURCE_DIR}/source/main.cpp)
  target_include_directories(${localname} PUBLIC ${simdir})
  target_include_directories(${localname} PUBLIC ${PROJECT_SOURCE_DIR}/source)
  target_link_libraries(${localname} PUBLIC ${localname}_lib)
endfunction(SIMULATION localname simdir)

function(CUDASIMCORELIB localname simdir)
  if (NOT TARGET ${localname})
    CUDA_INCLUDE_DIRECTORIES(${simdir} ${PROJECT_SOURCE_DIR}/source)
    configure_file(${PROJECT_SOURCE_DIR}/source/core/build_once.cpp ${PROJECT_SOURCE_DIR}/source/core/build_once.cu COPYONLY)
    cuda_add_library(${localname}
          ${PROJECT_SOURCE_DIR}/source/core/build_once.cu
          ${PROJECT_SOURCE_DIR}/source/sim/set_cuda.cu
          ${PROJECT_SOURCE_DIR}/source/sim/base.cpp
          ${PROJECT_SOURCE_DIR}/source/sim/determ/determ.cpp
          ${PROJECT_SOURCE_DIR}/source/sim/determ/determ_cuda.cu
          ${PROJECT_SOURCE_DIR}/source/sim/cell_param.cpp
          ${PROJECT_SOURCE_DIR}/source/sim/determ/baby_cl.cpp
          ${PROJECT_SOURCE_DIR}/source/sim/determ/baby_cl_cuda.cu
          ${PROJECT_SOURCE_DIR}/source/io/arg_parse.cpp
          ${PROJECT_SOURCE_DIR}/source/utility/common_utils.cpp
          ${PROJECT_SOURCE_DIR}/source/utility/color.cpp
          ${PROJECT_SOURCE_DIR}/source/utility/preprocessor.cpp
          ${PROJECT_SOURCE_DIR}/source/io/csvr.cpp
          ${PROJECT_SOURCE_DIR}/source/io/csvr_sim.cpp
          ${PROJECT_SOURCE_DIR}/source/io/csvw.cpp
          ${PROJECT_SOURCE_DIR}/source/io/csvw_sim.cpp
          ${PROJECT_SOURCE_DIR}/source/io/ezxml/ezxml.c)

    target_include_directories(${localname} PUBLIC ${simdir})
    target_include_directories(${localname} PUBLIC ${PROJECT_SOURCE_DIR}/source)
    target_link_libraries(${localname} ${CUDA_LIBRARIES})
  endif (NOT TARGET ${localname})
endfunction(CUDASIMCORELIB localname simdir)

function(CUDA_SIMULATION localname simdir)
  CUDASIMCORELIB(${localname}_lib ${simdir})
  CUDA_INCLUDE_DIRECTORIES(${simdir} ${PROJECT_SOURCE_DIR}/source)
  cuda_add_executable(${localname} ${PROJECT_SOURCE_DIR}/source/main_cuda.cu)
  target_include_directories(${localname} PUBLIC ${simdir})
  target_include_directories(${localname} PUBLIC ${PROJECT_SOURCE_DIR}/source)
  target_link_libraries(${localname} ${CUDA_LIBRARIES})
  target_link_libraries(${localname} ${localname}_lib)
endfunction(CUDA_SIMULATION localname simdir)

function(PARAM_SEARCH_LIB localname simdir)
  if (NOT TARGET ${localname})
    add_library(${localname}
                   ${PROJECT_SOURCE_DIR}/source/search/libsres/ESES.cpp
                   ${PROJECT_SOURCE_DIR}/source/search/libsres/ESSRSort.cpp
                   ${PROJECT_SOURCE_DIR}/source/search/libsres/sharefunc.cpp
                   ${PROJECT_SOURCE_DIR}/source/search/sres.cpp)
    target_include_directories(${localname} PUBLIC ${simdir})
    target_include_directories(${localname} PUBLIC ${PROJECT_SOURCE_DIR}/source)
  endif (NOT TARGET ${localname})
endfunction(PARAM_SEARCH_LIB localname simdir)


if( ${CMAKE_CURRENT_BINARY_DIR} STREQUAL ${PROJECT_BINARY_DIR}/source)
  set(BUILD_TESTS_ONLY FALSE)
  if (NOT EXISTS ${PROJECT_BINARY_DIR}/reaction_deltas.hpp)
    message("No reaction_deltas.cpp found in build directory")
    set(BUILD_TESTS_ONLY TRUE)
  endif (NOT EXISTS ${PROJECT_BINARY_DIR}/reaction_deltas.hpp)
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
    set_target_properties(simulation PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/binary")
    if (${CUDA_FOUND})
      CUDA_SIMULATION(cuda_simulation ${PROJECT_BINARY_DIR})
      set_target_properties(cuda_simulation PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/binary")
    endif (${CUDA_FOUND})
    # Forces csv_gen to run before all the test_'s are built.
    # This occurs when 'make' is run.
    add_custom_target(csv_gen_run ${PROJECT_BINARY_DIR}/source/csv_gen ${PROJECT_BINARY_DIR}/ )
    add_dependencies(csv_gen_run csv_gen)
    add_dependencies(simulation csv_gen_run)

    if (EXISTS ${PROJECT_BINARY_DIR}/param_search_main.cpp)
      PARAM_SEARCH(paramsearch_lib ${PROJECT_BINARY_DIR})
      add_executible(paramsearch ${PROJECT_BINARY_DIR}/param_search_main.cpp)
      target_link_libraries(paramsearch paramsearch_lib simulation_lib)
    endif (EXISTS ${PROJECT_BINARY_DIR}/param_search_main.cpp)
  endif (${BUILD_TESTS_ONLY})
#add_test(ctest_simulation main -p ../param_list.csv -c 200 -w 50 -s 0.01 -a 0.1 -r 10 -t 60 -e ../data_out.csv)
endif(${CMAKE_CURRENT_BINARY_DIR} STREQUAL ${PROJECT_BINARY_DIR}/source)
