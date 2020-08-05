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
          ${DENSE_SOURCE_DIR}/source/sim/determ/baby_cl.hpp
          ${DENSE_SOURCE_DIR}/source/sim/determ/simpson.cpp
          ${DENSE_SOURCE_DIR}/source/sim/determ/trap.cpp
          ${DENSE_SOURCE_DIR}/source/sim/determ/odepk.cpp
          ${DENSE_SOURCE_DIR}/source/sim/determ/num_sim.hpp
          ${DENSE_SOURCE_DIR}/source/sim/stoch/fast_gillespie_direct_simulation.cpp
          ${DENSE_SOURCE_DIR}/source/sim/stoch/next_reaction_simulation.cpp
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
          )
    target_include_directories(${localname} PUBLIC ${simdir})
    target_include_directories(${localname} PUBLIC ${DENSE_SOURCE_DIR}/source)
    target_include_directories(${localname} PUBLIC ${MPI_CXX_INCLUDE_PATH})
    target_link_libraries(${localname} ${MPI_CXX_LIBRARIES})
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

function(SIMPSON_SIMULATION localname simdir)
  SIMULATION(${localname} ${simdir} ${DENSE_SOURCE_DIR}/source/Simpson_main.cpp)
endfunction(SIMPSON_SIMULATION localname simdir)

function(TRAPEZOID_SIMULATION localname simdir)
  SIMULATION(${localname} ${simdir} ${DENSE_SOURCE_DIR}/source/Trapezoidal_main.cpp)
endfunction(TRAPEZOID_SIMULATION localname simdir)

function(ODEPACK_SIMULATION localname simdir)
  SIMULATION(${localname} ${simdir} ${DENSE_SOURCE_DIR}/source/ODEpk_main.cpp)
endfunction(ODEPACK_SIMULATION localname simdir)

function(GILLESPIE_SIMULATION localname simdir)
  SIMULATION(${localname} ${simdir} ${DENSE_SOURCE_DIR}/source/Fast_Gillespie_Main.cpp)
endfunction(GILLESPIE_SIMULATION localname simdir)

function(CELL_GROWTH localname simdir)
  SIMULATION(${localname} ${simdir} ${DENSE_SOURCE_DIR}/source/Cell_Grow_Main.cpp)
endfunction(CELL_GROWTH localname simdir)

function(STOCH_NR_SIMULATION localname simdir)
  SIMULATION(${localname} ${simdir} ${DENSE_SOURCE_DIR}/source/Next_Reaction_Main.cpp)
endfunction(STOCH_NR_SIMULATION localname simdir)

function(LOG_DATA_IMPORTS localname simdir)
  SIMULATION(${localname} ${simdir} ${DENSE_SOURCE_DIR}/source/log_data_imports.cpp)
endfunction(LOG_DATA_IMPORTS localname simdir)

function(IN_MEMORY_LOG localname simdir)
  SIMULATION(${localname} ${simdir} ${DENSE_SOURCE_DIR}/source/in_memory_log_main.cpp)
endfunction(IN_MEMORY_LOG localname simdir)

function(PARAMSEARCH localname simdir)
  SIMCORELIB(${localname}_lib ${simdir})
  add_executable(${localname} ${PROJECT_SOURCE_DIR}/source/param_search_main.cpp)
  target_include_directories(${localname} PUBLIC ${simdir})
  target_include_directories(${localname} PUBLIC ${PROJECT_SOURCE_DIR}/source)
  target_link_libraries(${localname} PUBLIC ${localname}_lib)
endfunction(PARAMSEARCH localname simdir)

function(PARAM_SEARCH_MPI_LIB localname simdir)
  if (NOT TARGET ${localname})
    add_library(${localname}
          ${DENSE_SOURCE_DIR}/source/core/build_once.cpp
          ${DENSE_SOURCE_DIR}/source/sim/base.cpp
          ${DENSE_SOURCE_DIR}/source/sim/determ/determ.cpp
          ${DENSE_SOURCE_DIR}/source/sim/determ/baby_cl.hpp
          ${DENSE_SOURCE_DIR}/source/sim/determ/simpson.cpp
          ${DENSE_SOURCE_DIR}/source/sim/determ/trap.cpp
          ${DENSE_SOURCE_DIR}/source/sim/determ/odepk.cpp
          ${DENSE_SOURCE_DIR}/source/sim/determ/num_sim.hpp
          ${DENSE_SOURCE_DIR}/source/sim/stoch/fast_gillespie_direct_simulation.cpp
          ${DENSE_SOURCE_DIR}/source/sim/stoch/next_reaction_simulation.cpp
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
	  ${DENSE_SOURCE_DIR}/source/run_analysis_only.hpp
          ${DENSE_SOURCE_DIR}/source/Sim_Builder.hpp
          ${DENSE_SOURCE_DIR}/source/search/sres.cpp
          ${DENSE_SOURCE_DIR}/source/search/libsres-mpi/ESES.cpp
          ${DENSE_SOURCE_DIR}/source/search/libsres-mpi/ESSRSort.cpp
          ${DENSE_SOURCE_DIR}/source/search/libsres-mpi/sharefunc.cpp
          )
    target_include_directories(${localname} PUBLIC ${simdir})
    target_include_directories(${localname} PUBLIC ${DENSE_SOURCE_DIR}/source)
    target_include_directories(${localname} PUBLIC ${MPI_CXX_INCLUDE_PATH})
    target_link_libraries(${localname} ${MPI_CXX_LIBRARIES})
  endif (NOT TARGET ${localname})
endfunction(PARAM_SEARCH_MPI_LIB localname simdir)

function(PARAMSEARCHMPI localname simdir)
  PARAM_SEARCH_MPI_LIB(${localname}_lib ${simdir})
  add_executable(${localname} ${PROJECT_SOURCE_DIR}/source/param_search_main.cpp)
  target_include_directories(${localname} PUBLIC ${simdir})
  target_include_directories(${localname} PUBLIC ${DENSE_SOURCE_DIR}/source)
  target_include_directories(${localname} PRIVATE ${MPI_CXX_INCLUDE_PATH})
  target_compile_options(${localname} PRIVATE ${MPI_CXX_COMPILE_FLAGS})
  target_link_libraries(${localname} ${MPI_CXX_LIBRARIES})
  target_link_libraries(${localname} ${localname}_lib)
endfunction(PARAMSEARCHMPI localname simdir)


function(CUDASIMCORELIB localname simdir)
  if (NOT TARGET ${localname})
    CUDA_INCLUDE_DIRECTORIES(${simdir} ${DENSE_SOURCE_DIR}/source)
    configure_file(${DENSE_SOURCE_DIR}/source/core/build_once.cpp ${DENSE_SOURCE_DIR}/source/core/build_once.cu COPYONLY)
    cuda_add_library(${localname}
          ${DENSE_SOURCE_DIR}/source/core/build_once.cu
          ${DENSE_SOURCE_DIR}/source/sim/set_cuda.cu
          ${DENSE_SOURCE_DIR}/source/sim/base.cpp
          ${DENSE_SOURCE_DIR}/source/sim/determ/num_sim.hpp
          ${DENSE_SOURCE_DIR}/source/sim/determ/simpson.cpp
          ${DENSE_SOURCE_DIR}/source/sim/determ/trap.cpp
          ${DENSE_SOURCE_DIR}/source/sim/determ/odepk.cpp
          ${DENSE_SOURCE_DIR}/source/sim/determ/determ.cpp
	  ${DENSE_SOURCE_DIR}/source/in_memory_log.hpp
          ${DENSE_SOURCE_DIR}/source/sim/determ/determ_cuda.cu
          ${DENSE_SOURCE_DIR}/source/sim/cell_param.cpp
          ${DENSE_SOURCE_DIR}/source/sim/determ/baby_cl.hpp
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
  if (${BUILD_TESTS_ONLY})
    message(WARNING "No or incomplete model implementation found - building test cases only")
  else (${BUILD_TESTS_ONLY})
    CSV_GEN(csv_gen ${PROJECT_BINARY_DIR})
    if (${DETERMINISTIC_SIM})
      DETERMINISTIC_SIMULATION(simulation ${PROJECT_BINARY_DIR})
      set_target_properties(simulation PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
    endif (${DETERMINISTIC_SIM})

    if (${SIMPSON_SIM})
      SIMPSON_SIMULATION(simpson_simulation ${PROJECT_BINARY_DIR})
      set_target_properties(simpson_simulation PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
    endif (${SIMPSON_SIM})

    if (${TRAPEZOID_SIM})
      TRAPEZOID_SIMULATION(trapezoid_simulation ${PROJECT_BINARY_DIR})
      set_target_properties(trapezoid_simulation PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
    endif (${TRAPEZOID_SIM})

    if (${ODEPACK_SIM})
      ODEPACK_SIMULATION(odepack_simulation ${PROJECT_BINARY_DIR})
      set_target_properties(odepack_simulation PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
    endif (${ODEPACK_SIM})

    if (${GILLESPIE_SIM})
      GILLESPIE_SIMULATION(gillespie ${PROJECT_BINARY_DIR})
      set_target_properties(gillespie PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
    endif (${GILLESPIE_SIM})

    if (${NEXTREACTION_SIM})
      STOCH_NR_SIMULATION(nextreaction ${PROJECT_BINARY_DIR})
      set_target_properties(nextreaction PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
    endif (${NEXTREACTION_SIM})
    if (${IN_MEM_LOG})
      IN_MEMORY_LOG(inmemorylog ${PROJECT_BINARY_DIR})
      set_target_properties(inmemorylog PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
    endif (${IN_MEM_LOG})


    if (${CUDA_FOUND})
      if (${CUDA_SIM})
        CUDA_SIMULATION(cuda_simulation ${PROJECT_BINARY_DIR})
        set_target_properties(cuda_simulation PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
      endif (${CUDA_SIM})
    endif (${CUDA_FOUND})
    if (${MPI_FOUND})
      if (${MPI_PARAM_SEARCH})
        PARAMSEARCHMPI(paramsearch_mpi ${PROJECT_BINARY_DIR})
        set_target_properties(paramsearch_mpi PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
      endif (${MPI_PARAM_SEARCH})
    else(${MPI_FOUND})
      if (${PARAM_SEARCH})
        PARAMSEARCH(paramsearch ${PROJECT_BINARY_DIR})
        set_target_properties(paramsearch PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
      endif (${PARAM_SEARCH})
    endif (${MPI_FOUND})
    # Forces csv_gen to run before all the test_'s are built.
    # This occurs when 'make' is run.
    add_custom_target(csv_gen_run ${PROJECT_BINARY_DIR}/source/csv_gen ${PROJECT_BINARY_DIR}/ )
    add_dependencies(csv_gen_run csv_gen)
    #add_dependencies(simulation csv_gen_run)
  endif (${BUILD_TESTS_ONLY})
endif(${CMAKE_CURRENT_BINARY_DIR} STREQUAL ${PROJECT_BINARY_DIR}/source)
