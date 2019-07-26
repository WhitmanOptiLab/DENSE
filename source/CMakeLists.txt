#include_directories("${CMAKE_CURRENT_SOURCE_DIR}")

function(target_link_simlibraries targetname simlibname)
  target_link_libraries(${targetname} ${simlibname})
  target_include_directories(${targetname} PUBLIC ${simlibname})
endfunction(target_link_simlibraries targetname simlibname)

function(CSV_GEN localname simdir)
  add_executable(${localname} 
          ${DENSE_SOURCE_DIR}/source/csv_gen.cpp 
          ${DENSE_SOURCE_DIR}/source/utility/color.cpp 
          ${DENSE_SOURCE_DIR}/source/io/csvw.cpp)
  target_include_directories(${localname} PUBLIC ${simdir})
  target_include_directories(${localname} PUBLIC ${DENSE_SOURCE_DIR}/source)
  target_include_directories(${localname} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
endfunction(CSV_GEN)

function(SIMCORELIB localname simdir)
  if (NOT TARGET ${localname})
    add_library(${localname}
          ${DENSE_SOURCE_DIR}/source/core/build_once.cpp
          ${DENSE_SOURCE_DIR}/source/sim/base.cpp
          ${DENSE_SOURCE_DIR}/source/sim/determ/determ.cpp
          ${DENSE_SOURCE_DIR}/source/sim/determ/baby_cl.cpp
          ${DENSE_SOURCE_DIR}/source/sim/stoch/fast_gillespie_direct_simulation.cpp
          ${DENSE_SOURCE_DIR}/source/sim/stoch/next_reaction_simulation.cpp
          ${DENSE_SOURCE_DIR}/source/sim/stoch/anderson_next_reaction_simulation.cpp
          ${DENSE_SOURCE_DIR}/source/sim/stoch/Gillespie_Direct_Simulation.cpp
          ${DENSE_SOURCE_DIR}/source/sim/stoch/stoch.cpp
          ${DENSE_SOURCE_DIR}/source/utility/common_utils.cpp
          ${DENSE_SOURCE_DIR}/source/utility/color.cpp
          ${DENSE_SOURCE_DIR}/source/utility/preprocessor.cpp
          ${DENSE_SOURCE_DIR}/source/io/arg_parse.cpp
          ${DENSE_SOURCE_DIR}/source/io/csvr.cpp
          ${DENSE_SOURCE_DIR}/source/io/csvr_sim.cpp
          ${DENSE_SOURCE_DIR}/source/io/csvw.cpp
          ${DENSE_SOURCE_DIR}/source/io/ezxml/ezxml.c
          ${DENSE_SOURCE_DIR}/source/arg_parse.hpp
          ${DENSE_SOURCE_DIR}/source/parse_analysis_entries.hpp
          ${DENSE_SOURCE_DIR}/source/run_simulation.hpp
          ${DENSE_SOURCE_DIR}/source/Sim_Builder.hpp
          ${DENSE_SOURCE_DIR}/source/Sim_Initializer.hpp
					${DENSE_SOURCE_DIR}/source/search/sres.cpp
					${DENSE_SOURCE_DIR}/source/search/libsres/ESES.cpp
					${DENSE_SOURCE_DIR}/source/search/libsres/ESSRSort.cpp
					${DENSE_SOURCE_DIR}/source/search/libsres/sharefunc.cpp
          ${DENSE_SOURCE_DIR}/source/sim/stoch/propensity_groups.hpp
          ${DENSE_SOURCE_DIR}/source/sim/stoch/rejection_based_simulation.cpp
          ${DENSE_SOURCE_DIR}/source/rejection_based_main.cpp
          ${DENSE_SOURCE_DIR}/source/sim/stoch/rxn_struct.hpp

          )
    target_include_directories(${localname} PUBLIC ${simdir})
    target_include_directories(${localname} PUBLIC ${DENSE_SOURCE_DIR}/source)
  endif (NOT TARGET ${localname})
endfunction(SIMCORELIB localname simdir)

function(SIMULATION localname simdir mainfile)
  SIMCORELIB(${localname}_lib ${simdir})
  add_executable(${localname} ${mainfile})
  target_include_directories(${localname} PUBLIC ${simdir})
  target_include_directories(${localname} PUBLIC ${DENSE_SOURCE_DIR}/source)
  target_link_libraries(${localname} PUBLIC ${localname}_lib)
endfunction(SIMULATION localname simdir mainfile)

function(DETERMINISTIC_SIMULATION localname simdir)
  SIMULATION(${localname} ${simdir} ${DENSE_SOURCE_DIR}/source/deterministic_main.cpp)
endfunction(DETERMINISTIC_SIMULATION localname simdir)

function(REJECTION_BASED_SIMULATION localname simdir)
  SIMULATION(${localname} ${simdir} ${DENSE_SOURCE_DIR}/source/rejection_based_main.cpp)
endfunction(REJECTION_BASED_SIMULATION localname simdir)

function(GILLESPIE_SIMULATION localname simdir)
  SIMULATION(${localname} ${simdir} ${DENSE_SOURCE_DIR}/source/Gillespie_Direct_main.cpp)
endfunction(GILLESPIE_SIMULATION localname simdir)

function(ANDERSON_SIMULATION localname simdir)
  SIMULATION(${localname} ${simdir} ${DENSE_SOURCE_DIR}/source/Anderson_main.cpp)
endfunction(ANDERSON_SIMULATION localname simdir)

 function(FAST_GILLESPIE_SIMULATION localname simdir)
  SIMULATION(${localname} ${simdir} ${DENSE_SOURCE_DIR}/source/Fast_Gillespie_Main.cpp)
endfunction(FAST_GILLESPIE_SIMULATION localname simdir)

function(STOCH_NR_SIMULATION localname simdir)
  SIMULATION(${localname} ${simdir} ${DENSE_SOURCE_DIR}/source/Next_Reaction_Main.cpp)
endfunction(STOCH_NR_SIMULATION localname simdir)


#function(PARAMSEARCH localname simdir)
#  SIMCORELIB(${localname}_lib ${simdir})
#  add_executable(${localname} ${PROJECT_SOURCE_DIR}/source/param_search_main.cpp)
#  target_include_directories(${localname} PUBLIC ${simdir})
#  target_include_directories(${localname} PUBLIC ${PROJECT_SOURCE_DIR}/source)
#  target_link_libraries(${localname} PUBLIC ${localname}_lib)
#endfunction(PARAMSEARCH localname simdir)



function(CUDASIMCORELIB localname simdir)
  if (NOT TARGET ${localname})
    CUDA_INCLUDE_DIRECTORIES(${simdir} ${DENSE_SOURCE_DIR}/source)
    configure_file(${DENSE_SOURCE_DIR}/source/core/build_once.cpp ${DENSE_SOURCE_DIR}/source/core/build_once.cu COPYONLY)
    cuda_add_library(${localname}
          ${DENSE_SOURCE_DIR}/source/core/build_once.cu
          ${DENSE_SOURCE_DIR}/source/sim/set_cuda.cu
          ${DENSE_SOURCE_DIR}/source/sim/base.cpp
          ${DENSE_SOURCE_DIR}/source/sim/determ/determ.cpp
          ${DENSE_SOURCE_DIR}/source/sim/determ/determ_cuda.cu
          ${DENSE_SOURCE_DIR}/source/sim/cell_param.cpp
          ${DENSE_SOURCE_DIR}/source/sim/determ/baby_cl.cpp
          ${DENSE_SOURCE_DIR}/source/sim/determ/baby_cl_cuda.cu
          ${DENSE_SOURCE_DIR}/source/io/arg_parse.cpp
          ${DENSE_SOURCE_DIR}/source/utility/common_utils.cpp
          ${DENSE_SOURCE_DIR}/source/utility/color.cpp
          ${DENSE_SOURCE_DIR}/source/utility/preprocessor.cpp
          ${DENSE_SOURCE_DIR}/source/io/csvr.cpp
          ${DENSE_SOURCE_DIR}/source/io/csvr_sim.cpp
          ${DENSE_SOURCE_DIR}/source/io/csvw.cpp
          ${DENSE_SOURCE_DIR}/source/io/csvw_sim.cpp
          ${DENSE_SOURCE_DIR}/source/io/ezxml/ezxml.c)

    target_include_directories(${localname} PUBLIC ${simdir})
    target_include_directories(${localname} PUBLIC ${DENSE_SOURCE_DIR}/source)
    target_link_libraries(${localname} ${CUDA_LIBRARIES})
  endif (NOT TARGET ${localname})
endfunction(CUDASIMCORELIB localname simdir)

function(CUDA_SIMULATION localname simdir)
  CUDASIMCORELIB(${localname}_lib ${simdir})
  CUDA_INCLUDE_DIRECTORIES(${simdir} ${DENSE_SOURCE_DIR}/source)
  cuda_add_executable(${localname} ${DENSE_SOURCE_DIR}/source/main_cuda.cu)
  target_include_directories(${localname} PUBLIC ${simdir})
  target_include_directories(${localname} PUBLIC ${DENSE_SOURCE_DIR}/source)
  target_link_libraries(${localname} ${CUDA_LIBRARIES})
  target_link_libraries(${localname} ${localname}_lib)
endfunction(CUDA_SIMULATION localname simdir)

function(PARAM_SEARCH_LIB localname simdir)
  if (NOT TARGET ${localname})
    add_library(${localname}
                   ${DENSE_SOURCE_DIR}/source/search/libsres/ESES.cpp
                   ${DENSE_SOURCE_DIR}/source/search/libsres/ESSRSort.cpp
                   ${DENSE_SOURCE_DIR}/source/search/libsres/sharefunc.cpp
                   ${DENSE_SOURCE_DIR}/source/search/sres.cpp)
    target_include_directories(${localname} PUBLIC ${simdir})
    target_include_directories(${localname} PUBLIC ${DENSE_SOURCE_DIR}/source)
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
    DETERMINISTIC_SIMULATION(simulation ${PROJECT_BINARY_DIR})
    set_target_properties(simulation PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
    REJECTION_BASED_SIMULATION(rejection ${PROJECT_BINARY_DIR})
    set_target_properties(rejection PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
    FAST_GILLESPIE_SIMULATION(fast_gillespie ${PROJECT_BINARY_DIR})
    set_target_properties(fast_gillespie PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
    STOCH_NR_SIMULATION(nextreaction ${PROJECT_BINARY_DIR})
    set_target_properties(nextreaction PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
    ANDERSON_SIMULATION(anderson ${PROJECT_BINARY_DIR})
    set_target_properties(anderson PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
    GILLESPIE_SIMULATION(gillespie ${PROJECT_BINARY_DIR})
    set_target_properties(gillespie PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
    if (${CUDA_FOUND})
      CUDA_SIMULATION(cuda_simulation ${PROJECT_BINARY_DIR})
      set_target_properties(cuda_simulation PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
    endif (${CUDA_FOUND})
    # Forces csv_gen to run before all the test_'s are built.
    # This occurs when 'make' is run.
    add_custom_target(csv_gen_run ${PROJECT_BINARY_DIR}/source/csv_gen ${PROJECT_BINARY_DIR}/ )
    add_dependencies(csv_gen_run csv_gen)
    add_dependencies(simulation csv_gen_run)
  endif (${BUILD_TESTS_ONLY})
endif(${CMAKE_CURRENT_BINARY_DIR} STREQUAL ${PROJECT_BINARY_DIR}/source)