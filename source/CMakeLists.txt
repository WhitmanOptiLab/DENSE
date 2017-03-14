cmake_minimum_required(VERSION 3.6)
project(source)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES
        analysis.hpp
        baby_cl.cpp
        baby_cl.hpp
        cell_param.cpp
        cell_param.hpp
        concentration_level.cpp
        concentration_level.hpp
        context.cpp
        context.hpp
        experiment.hpp
        feature.hpp
        model.cpp
        model.hpp
        model_impl.hpp
        param_set.hpp
        reaction.cpp
        reaction.hpp
        reactions_list.hpp
        simulation.cpp
        simulation.hpp
        specie.hpp
        specie_list.hpp
        temp_test.cpp
        test_sim_compile.cpp
        test_simulation.cpp)

add_executable(source ${SOURCE_FILES})