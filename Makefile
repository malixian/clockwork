# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Default target executed when no arguments are given to make.
default_target: all

.PHONY : default_target

# Allow only one "make -f Makefile2" at a time, but pass parallelism.
.NOTPARALLEL:


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
CMAKE_SOURCE_DIR = /home/rkarimi/projects/clockwork

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/rkarimi/projects/clockwork

#=============================================================================
# Targets provided globally by CMake.

# Special rule for the target rebuild_cache
rebuild_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake to regenerate build system..."
	/usr/bin/cmake -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : rebuild_cache

# Special rule for the target rebuild_cache
rebuild_cache/fast: rebuild_cache

.PHONY : rebuild_cache/fast

# Special rule for the target edit_cache
edit_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "No interactive CMake dialog available..."
	/usr/bin/cmake -E echo No\ interactive\ CMake\ dialog\ available.
.PHONY : edit_cache

# Special rule for the target edit_cache
edit_cache/fast: edit_cache

.PHONY : edit_cache/fast

# The main all target
all: cmake_check_build_system
	$(CMAKE_COMMAND) -E cmake_progress_start /home/rkarimi/projects/clockwork/CMakeFiles /home/rkarimi/projects/clockwork/CMakeFiles/progress.marks
	$(MAKE) -f CMakeFiles/Makefile2 all
	$(CMAKE_COMMAND) -E cmake_progress_start /home/rkarimi/projects/clockwork/CMakeFiles 0
.PHONY : all

# The main clean target
clean:
	$(MAKE) -f CMakeFiles/Makefile2 clean
.PHONY : clean

# The main clean target
clean/fast: clean

.PHONY : clean/fast

# Prepare targets for installation.
preinstall: all
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall

# Prepare targets for installation.
preinstall/fast:
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall/fast

# clear depends
depend:
	$(CMAKE_COMMAND) -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 1
.PHONY : depend

#=============================================================================
# Target rules for targets named multitenant

# Build rule for target.
multitenant: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 multitenant
.PHONY : multitenant

# fast build rule for target.
multitenant/fast:
	$(MAKE) -f CMakeFiles/multitenant.dir/build.make CMakeFiles/multitenant.dir/build
.PHONY : multitenant/fast

#=============================================================================
# Target rules for targets named main

# Build rule for target.
main: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 main
.PHONY : main

# fast build rule for target.
main/fast:
	$(MAKE) -f CMakeFiles/main.dir/build.make CMakeFiles/main.dir/build
.PHONY : main/fast

src/clockwork/clockwork_queue.o: src/clockwork/clockwork_queue.cpp.o

.PHONY : src/clockwork/clockwork_queue.o

# target to build an object file
src/clockwork/clockwork_queue.cpp.o:
	$(MAKE) -f CMakeFiles/main.dir/build.make CMakeFiles/main.dir/src/clockwork/clockwork_queue.cpp.o
.PHONY : src/clockwork/clockwork_queue.cpp.o

src/clockwork/clockwork_queue.i: src/clockwork/clockwork_queue.cpp.i

.PHONY : src/clockwork/clockwork_queue.i

# target to preprocess a source file
src/clockwork/clockwork_queue.cpp.i:
	$(MAKE) -f CMakeFiles/main.dir/build.make CMakeFiles/main.dir/src/clockwork/clockwork_queue.cpp.i
.PHONY : src/clockwork/clockwork_queue.cpp.i

src/clockwork/clockwork_queue.s: src/clockwork/clockwork_queue.cpp.s

.PHONY : src/clockwork/clockwork_queue.s

# target to generate assembly for a file
src/clockwork/clockwork_queue.cpp.s:
	$(MAKE) -f CMakeFiles/main.dir/build.make CMakeFiles/main.dir/src/clockwork/clockwork_queue.cpp.s
.PHONY : src/clockwork/clockwork_queue.cpp.s

src/clockwork/multitenant/executor.o: src/clockwork/multitenant/executor.cc.o

.PHONY : src/clockwork/multitenant/executor.o

# target to build an object file
src/clockwork/multitenant/executor.cc.o:
	$(MAKE) -f CMakeFiles/multitenant.dir/build.make CMakeFiles/multitenant.dir/src/clockwork/multitenant/executor.cc.o
.PHONY : src/clockwork/multitenant/executor.cc.o

src/clockwork/multitenant/executor.i: src/clockwork/multitenant/executor.cc.i

.PHONY : src/clockwork/multitenant/executor.i

# target to preprocess a source file
src/clockwork/multitenant/executor.cc.i:
	$(MAKE) -f CMakeFiles/multitenant.dir/build.make CMakeFiles/multitenant.dir/src/clockwork/multitenant/executor.cc.i
.PHONY : src/clockwork/multitenant/executor.cc.i

src/clockwork/multitenant/executor.s: src/clockwork/multitenant/executor.cc.s

.PHONY : src/clockwork/multitenant/executor.s

# target to generate assembly for a file
src/clockwork/multitenant/executor.cc.s:
	$(MAKE) -f CMakeFiles/multitenant.dir/build.make CMakeFiles/multitenant.dir/src/clockwork/multitenant/executor.cc.s
.PHONY : src/clockwork/multitenant/executor.cc.s

src/clockwork/multitenant/manager.o: src/clockwork/multitenant/manager.cc.o

.PHONY : src/clockwork/multitenant/manager.o

# target to build an object file
src/clockwork/multitenant/manager.cc.o:
	$(MAKE) -f CMakeFiles/multitenant.dir/build.make CMakeFiles/multitenant.dir/src/clockwork/multitenant/manager.cc.o
.PHONY : src/clockwork/multitenant/manager.cc.o

src/clockwork/multitenant/manager.i: src/clockwork/multitenant/manager.cc.i

.PHONY : src/clockwork/multitenant/manager.i

# target to preprocess a source file
src/clockwork/multitenant/manager.cc.i:
	$(MAKE) -f CMakeFiles/multitenant.dir/build.make CMakeFiles/multitenant.dir/src/clockwork/multitenant/manager.cc.i
.PHONY : src/clockwork/multitenant/manager.cc.i

src/clockwork/multitenant/manager.s: src/clockwork/multitenant/manager.cc.s

.PHONY : src/clockwork/multitenant/manager.s

# target to generate assembly for a file
src/clockwork/multitenant/manager.cc.s:
	$(MAKE) -f CMakeFiles/multitenant.dir/build.make CMakeFiles/multitenant.dir/src/clockwork/multitenant/manager.cc.s
.PHONY : src/clockwork/multitenant/manager.cc.s

src/main.o: src/main.cc.o

.PHONY : src/main.o

# target to build an object file
src/main.cc.o:
	$(MAKE) -f CMakeFiles/main.dir/build.make CMakeFiles/main.dir/src/main.cc.o
.PHONY : src/main.cc.o

src/main.i: src/main.cc.i

.PHONY : src/main.i

# target to preprocess a source file
src/main.cc.i:
	$(MAKE) -f CMakeFiles/main.dir/build.make CMakeFiles/main.dir/src/main.cc.i
.PHONY : src/main.cc.i

src/main.s: src/main.cc.s

.PHONY : src/main.s

# target to generate assembly for a file
src/main.cc.s:
	$(MAKE) -f CMakeFiles/main.dir/build.make CMakeFiles/main.dir/src/main.cc.s
.PHONY : src/main.cc.s

# Help Target
help:
	@echo "The following are some of the valid targets for this Makefile:"
	@echo "... all (the default if no target is provided)"
	@echo "... clean"
	@echo "... depend"
	@echo "... rebuild_cache"
	@echo "... multitenant"
	@echo "... edit_cache"
	@echo "... main"
	@echo "... src/clockwork/clockwork_queue.o"
	@echo "... src/clockwork/clockwork_queue.i"
	@echo "... src/clockwork/clockwork_queue.s"
	@echo "... src/clockwork/multitenant/executor.o"
	@echo "... src/clockwork/multitenant/executor.i"
	@echo "... src/clockwork/multitenant/executor.s"
	@echo "... src/clockwork/multitenant/manager.o"
	@echo "... src/clockwork/multitenant/manager.i"
	@echo "... src/clockwork/multitenant/manager.s"
	@echo "... src/main.o"
	@echo "... src/main.i"
	@echo "... src/main.s"
.PHONY : help



#=============================================================================
# Special targets to cleanup operation of make.

# Special rule to run CMake to check the build system integrity.
# No rule that depends on this can have commands that come from listfiles
# because they might be regenerated.
cmake_check_build_system:
	$(CMAKE_COMMAND) -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 0
.PHONY : cmake_check_build_system

