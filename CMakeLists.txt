cmake_minimum_required (VERSION 2.6)
project (DDESimulatorTemplate CXX)
add_compile_options(-std=c++11)
enable_testing()

include_directories("${PROJECT_BINARY_DIR}")

add_subdirectory(source)
