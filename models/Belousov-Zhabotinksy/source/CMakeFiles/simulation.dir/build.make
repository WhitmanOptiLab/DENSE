# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/mcclelnr/DENSE

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/mcclelnr/DENSE/models/Belousov-Zhabotinksy

# Include any dependencies generated for this target.
include source/CMakeFiles/simulation.dir/depend.make

# Include the progress variables for this target.
include source/CMakeFiles/simulation.dir/progress.make

# Include the compile flags for this target's objects.
include source/CMakeFiles/simulation.dir/flags.make

source/CMakeFiles/simulation.dir/main.cpp.o: source/CMakeFiles/simulation.dir/flags.make
source/CMakeFiles/simulation.dir/main.cpp.o: ../../source/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/mcclelnr/DENSE/models/Belousov-Zhabotinksy/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object source/CMakeFiles/simulation.dir/main.cpp.o"
	cd /home/mcclelnr/DENSE/models/Belousov-Zhabotinksy/source && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/simulation.dir/main.cpp.o -c /home/mcclelnr/DENSE/source/main.cpp

source/CMakeFiles/simulation.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/simulation.dir/main.cpp.i"
	cd /home/mcclelnr/DENSE/models/Belousov-Zhabotinksy/source && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/mcclelnr/DENSE/source/main.cpp > CMakeFiles/simulation.dir/main.cpp.i

source/CMakeFiles/simulation.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/simulation.dir/main.cpp.s"
	cd /home/mcclelnr/DENSE/models/Belousov-Zhabotinksy/source && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/mcclelnr/DENSE/source/main.cpp -o CMakeFiles/simulation.dir/main.cpp.s

source/CMakeFiles/simulation.dir/main.cpp.o.requires:

.PHONY : source/CMakeFiles/simulation.dir/main.cpp.o.requires

source/CMakeFiles/simulation.dir/main.cpp.o.provides: source/CMakeFiles/simulation.dir/main.cpp.o.requires
	$(MAKE) -f source/CMakeFiles/simulation.dir/build.make source/CMakeFiles/simulation.dir/main.cpp.o.provides.build
.PHONY : source/CMakeFiles/simulation.dir/main.cpp.o.provides

source/CMakeFiles/simulation.dir/main.cpp.o.provides.build: source/CMakeFiles/simulation.dir/main.cpp.o


# Object files for target simulation
simulation_OBJECTS = \
"CMakeFiles/simulation.dir/main.cpp.o"

# External object files for target simulation
simulation_EXTERNAL_OBJECTS =

simulation: source/CMakeFiles/simulation.dir/main.cpp.o
simulation: source/CMakeFiles/simulation.dir/build.make
simulation: source/libsimulation_lib.a
simulation: source/CMakeFiles/simulation.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/mcclelnr/DENSE/models/Belousov-Zhabotinksy/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../simulation"
	cd /home/mcclelnr/DENSE/models/Belousov-Zhabotinksy/source && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/simulation.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
source/CMakeFiles/simulation.dir/build: simulation

.PHONY : source/CMakeFiles/simulation.dir/build

source/CMakeFiles/simulation.dir/requires: source/CMakeFiles/simulation.dir/main.cpp.o.requires

.PHONY : source/CMakeFiles/simulation.dir/requires

source/CMakeFiles/simulation.dir/clean:
	cd /home/mcclelnr/DENSE/models/Belousov-Zhabotinksy/source && $(CMAKE_COMMAND) -P CMakeFiles/simulation.dir/cmake_clean.cmake
.PHONY : source/CMakeFiles/simulation.dir/clean

source/CMakeFiles/simulation.dir/depend:
	cd /home/mcclelnr/DENSE/models/Belousov-Zhabotinksy && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/mcclelnr/DENSE /home/mcclelnr/DENSE/source /home/mcclelnr/DENSE/models/Belousov-Zhabotinksy /home/mcclelnr/DENSE/models/Belousov-Zhabotinksy/source /home/mcclelnr/DENSE/models/Belousov-Zhabotinksy/source/CMakeFiles/simulation.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : source/CMakeFiles/simulation.dir/depend

