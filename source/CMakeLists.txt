include_directories("${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(csv_gen csv_gen.cpp color.cpp csvw.cpp csvw_param.cpp)
add_executable(test_simulation test_simulation.cpp simulation_base.cpp simulation_determ.cpp simulation_stoch.cpp ${PROJECT_BINARY_DIR}/reaction.cpp cell_param.cpp baby_cl.cpp arg_parse.cpp model.cpp color.cpp observable.cpp analysis.cpp csvr.cpp csvr_param.cpp csvr_sim.cpp csvw.cpp csvw_param.cpp csvw_sim.cpp)
add_executable(test_simulation_set test_simulation_set.cpp simulation_base.cpp simulation_determ.cpp simulation_stoch.cpp ${PROJECT_BINARY_DIR}/reaction.cpp cell_param.cpp baby_cl.cpp arg_parse.cpp model.cpp color.cpp observable.cpp csvr.cpp csvr_param.cpp csvr_sim.cpp csvw.cpp csvw_param.cpp csvw_sim.cpp)
add_executable(test_main test_main.cpp simulation_base.cpp simulation_determ.cpp simulation_stoch.cpp ${PROJECT_BINARY_DIR}/reaction.cpp cell_param.cpp baby_cl.cpp arg_parse.cpp model.cpp color.cpp observable.cpp analysis.cpp csvr.cpp csvr_param.cpp csvr_sim.cpp csvw.cpp csvw_param.cpp csvw_sim.cpp)

add_test(ctest_simulation test_simulation -p ../param_list.csv)
add_test(ctest_simulation_set test_simulation_set -p ../param_list.csv)
add_test(ctest_main test_main -p ../param_list.csv -c 200 -w 50 -s 0.01 -a 0.1 -r 10 -t 60 -e ../data_out.csv)



# Forces csv_gen to run before all the test_'s are built.
# This occurs when 'make' is run.
add_custom_target(csv_gen_run ${PROJECT_BINARY_DIR}/source/csv_gen ../)

add_dependencies(test_simulation csv_gen_run)
add_dependencies(test_simulation_set csv_gen_run)
add_dependencies(test_main csv_gen_run)



# Warns user about lack of param_list.csv
if(NOT EXISTS "param_list.csv")
    message(WARNING "Upon running 'make', if it doesn't already exist, 'param_list_template.csv' will be generated in the model directory. Please fill out this template then rename it to 'param_list.csv' before attempting to run any compiled tests or executables.")
endif()
