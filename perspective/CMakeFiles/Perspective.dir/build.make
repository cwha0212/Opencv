# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

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
CMAKE_SOURCE_DIR = /home/cwha0212/Opencv_test/perspective

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/cwha0212/Opencv_test/perspective

# Include any dependencies generated for this target.
include CMakeFiles/Perspective.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/Perspective.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/Perspective.dir/flags.make

CMakeFiles/Perspective.dir/main.cpp.o: CMakeFiles/Perspective.dir/flags.make
CMakeFiles/Perspective.dir/main.cpp.o: main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/cwha0212/Opencv_test/perspective/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/Perspective.dir/main.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Perspective.dir/main.cpp.o -c /home/cwha0212/Opencv_test/perspective/main.cpp

CMakeFiles/Perspective.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Perspective.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/cwha0212/Opencv_test/perspective/main.cpp > CMakeFiles/Perspective.dir/main.cpp.i

CMakeFiles/Perspective.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Perspective.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/cwha0212/Opencv_test/perspective/main.cpp -o CMakeFiles/Perspective.dir/main.cpp.s

# Object files for target Perspective
Perspective_OBJECTS = \
"CMakeFiles/Perspective.dir/main.cpp.o"

# External object files for target Perspective
Perspective_EXTERNAL_OBJECTS =

Perspective: CMakeFiles/Perspective.dir/main.cpp.o
Perspective: CMakeFiles/Perspective.dir/build.make
Perspective: /usr/local/lib/libopencv_dnn.so.4.0.0
Perspective: /usr/local/lib/libopencv_gapi.so.4.0.0
Perspective: /usr/local/lib/libopencv_ml.so.4.0.0
Perspective: /usr/local/lib/libopencv_objdetect.so.4.0.0
Perspective: /usr/local/lib/libopencv_photo.so.4.0.0
Perspective: /usr/local/lib/libopencv_stitching.so.4.0.0
Perspective: /usr/local/lib/libopencv_video.so.4.0.0
Perspective: /usr/local/lib/libopencv_calib3d.so.4.0.0
Perspective: /usr/local/lib/libopencv_features2d.so.4.0.0
Perspective: /usr/local/lib/libopencv_flann.so.4.0.0
Perspective: /usr/local/lib/libopencv_highgui.so.4.0.0
Perspective: /usr/local/lib/libopencv_videoio.so.4.0.0
Perspective: /usr/local/lib/libopencv_imgcodecs.so.4.0.0
Perspective: /usr/local/lib/libopencv_imgproc.so.4.0.0
Perspective: /usr/local/lib/libopencv_core.so.4.0.0
Perspective: CMakeFiles/Perspective.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/cwha0212/Opencv_test/perspective/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable Perspective"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Perspective.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/Perspective.dir/build: Perspective

.PHONY : CMakeFiles/Perspective.dir/build

CMakeFiles/Perspective.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/Perspective.dir/cmake_clean.cmake
.PHONY : CMakeFiles/Perspective.dir/clean

CMakeFiles/Perspective.dir/depend:
	cd /home/cwha0212/Opencv_test/perspective && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/cwha0212/Opencv_test/perspective /home/cwha0212/Opencv_test/perspective /home/cwha0212/Opencv_test/perspective /home/cwha0212/Opencv_test/perspective /home/cwha0212/Opencv_test/perspective/CMakeFiles/Perspective.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/Perspective.dir/depend
