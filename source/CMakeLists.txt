include_directories("${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(gen_csv gen_csv.cpp)
add_executable(test_sim_compile test_sim_compile.cpp simulation.cpp baby_cl.cpp cell_param.cpp model.cpp ${PROJECT_BINARY_DIR}/reaction.cpp)
add_executable(test_simulation test_simulation.cpp simulation.cpp ${PROJECT_BINARY_DIR}/reaction.cpp cell_param.cpp baby_cl.cpp param_set.cpp arg_parse.cpp model.cpp)
add_executable(test_simulation_set test_simulation_set.cpp simulation.cpp ${PROJECT_BINARY_DIR}/reaction.cpp cell_param.cpp baby_cl.cpp param_set.cpp arg_parse.cpp model.cpp)

