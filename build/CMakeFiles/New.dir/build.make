# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

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
CMAKE_SOURCE_DIR = /home/alessio/dev/BasketFin

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/alessio/dev/BasketFin/build

# Include any dependencies generated for this target.
include CMakeFiles/New.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/New.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/New.dir/flags.make

CMakeFiles/New.dir/new_planes.cpp.o: CMakeFiles/New.dir/flags.make
CMakeFiles/New.dir/new_planes.cpp.o: ../new_planes.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/alessio/dev/BasketFin/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/New.dir/new_planes.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/New.dir/new_planes.cpp.o -c /home/alessio/dev/BasketFin/new_planes.cpp

CMakeFiles/New.dir/new_planes.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/New.dir/new_planes.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/alessio/dev/BasketFin/new_planes.cpp > CMakeFiles/New.dir/new_planes.cpp.i

CMakeFiles/New.dir/new_planes.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/New.dir/new_planes.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/alessio/dev/BasketFin/new_planes.cpp -o CMakeFiles/New.dir/new_planes.cpp.s

CMakeFiles/New.dir/new_planes.cpp.o.requires:

.PHONY : CMakeFiles/New.dir/new_planes.cpp.o.requires

CMakeFiles/New.dir/new_planes.cpp.o.provides: CMakeFiles/New.dir/new_planes.cpp.o.requires
	$(MAKE) -f CMakeFiles/New.dir/build.make CMakeFiles/New.dir/new_planes.cpp.o.provides.build
.PHONY : CMakeFiles/New.dir/new_planes.cpp.o.provides

CMakeFiles/New.dir/new_planes.cpp.o.provides.build: CMakeFiles/New.dir/new_planes.cpp.o


CMakeFiles/New.dir/classes/newBasketModel.cpp.o: CMakeFiles/New.dir/flags.make
CMakeFiles/New.dir/classes/newBasketModel.cpp.o: ../classes/newBasketModel.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/alessio/dev/BasketFin/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/New.dir/classes/newBasketModel.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/New.dir/classes/newBasketModel.cpp.o -c /home/alessio/dev/BasketFin/classes/newBasketModel.cpp

CMakeFiles/New.dir/classes/newBasketModel.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/New.dir/classes/newBasketModel.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/alessio/dev/BasketFin/classes/newBasketModel.cpp > CMakeFiles/New.dir/classes/newBasketModel.cpp.i

CMakeFiles/New.dir/classes/newBasketModel.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/New.dir/classes/newBasketModel.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/alessio/dev/BasketFin/classes/newBasketModel.cpp -o CMakeFiles/New.dir/classes/newBasketModel.cpp.s

CMakeFiles/New.dir/classes/newBasketModel.cpp.o.requires:

.PHONY : CMakeFiles/New.dir/classes/newBasketModel.cpp.o.requires

CMakeFiles/New.dir/classes/newBasketModel.cpp.o.provides: CMakeFiles/New.dir/classes/newBasketModel.cpp.o.requires
	$(MAKE) -f CMakeFiles/New.dir/build.make CMakeFiles/New.dir/classes/newBasketModel.cpp.o.provides.build
.PHONY : CMakeFiles/New.dir/classes/newBasketModel.cpp.o.provides

CMakeFiles/New.dir/classes/newBasketModel.cpp.o.provides.build: CMakeFiles/New.dir/classes/newBasketModel.cpp.o


# Object files for target New
New_OBJECTS = \
"CMakeFiles/New.dir/new_planes.cpp.o" \
"CMakeFiles/New.dir/classes/newBasketModel.cpp.o"

# External object files for target New
New_EXTERNAL_OBJECTS =

New: CMakeFiles/New.dir/new_planes.cpp.o
New: CMakeFiles/New.dir/classes/newBasketModel.cpp.o
New: CMakeFiles/New.dir/build.make
New: /usr/local/lib/libpcl_stereo.so
New: /usr/local/lib/libpcl_surface.so
New: /usr/local/lib/libpcl_recognition.so
New: /usr/local/lib/libpcl_keypoints.so
New: /usr/local/lib/libpcl_people.so
New: /usr/local/lib/libpcl_tracking.so
New: /usr/local/lib/libpcl_outofcore.so
New: /usr/lib/x86_64-linux-gnu/libboost_system.so
New: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
New: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
New: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
New: /usr/lib/x86_64-linux-gnu/libboost_serialization.so
New: /usr/lib/x86_64-linux-gnu/libboost_regex.so
New: /usr/lib/x86_64-linux-gnu/libqhull.so
New: /usr/lib/x86_64-linux-gnu/libflann_cpp.so
New: /usr/lib/x86_64-linux-gnu/libboost_system.so
New: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
New: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
New: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
New: /usr/lib/x86_64-linux-gnu/libboost_serialization.so
New: /usr/lib/x86_64-linux-gnu/libboost_regex.so
New: /usr/local/lib/libpcl_registration.so
New: /usr/local/lib/libpcl_segmentation.so
New: /usr/local/lib/libpcl_features.so
New: /usr/local/lib/libvtkChartsCore-7.1.so.1
New: /usr/local/lib/libvtkInfovisCore-7.1.so.1
New: /usr/local/lib/libvtkIOGeometry-7.1.so.1
New: /usr/local/lib/libvtkIOLegacy-7.1.so.1
New: /usr/local/lib/libvtkIOPLY-7.1.so.1
New: /usr/local/lib/libvtkRenderingLOD-7.1.so.1
New: /usr/local/lib/libvtkViewsContext2D-7.1.so.1
New: /usr/local/lib/libvtkViewsCore-7.1.so.1
New: /usr/local/lib/libvtkInteractionWidgets-7.1.so.1
New: /usr/local/lib/libvtkFiltersModeling-7.1.so.1
New: /usr/local/lib/libvtkInteractionStyle-7.1.so.1
New: /usr/local/lib/libvtkFiltersExtraction-7.1.so.1
New: /usr/local/lib/libvtkFiltersStatistics-7.1.so.1
New: /usr/local/lib/libvtkImagingFourier-7.1.so.1
New: /usr/local/lib/libvtkalglib-7.1.so.1
New: /usr/local/lib/libvtkFiltersHybrid-7.1.so.1
New: /usr/local/lib/libvtkImagingGeneral-7.1.so.1
New: /usr/local/lib/libvtkImagingSources-7.1.so.1
New: /usr/local/lib/libvtkImagingHybrid-7.1.so.1
New: /usr/local/lib/libvtkRenderingAnnotation-7.1.so.1
New: /usr/local/lib/libvtkImagingColor-7.1.so.1
New: /usr/local/lib/libvtkRenderingVolume-7.1.so.1
New: /usr/local/lib/libvtkIOXML-7.1.so.1
New: /usr/local/lib/libvtkIOXMLParser-7.1.so.1
New: /usr/local/lib/libvtkIOCore-7.1.so.1
New: /usr/local/lib/libvtkexpat-7.1.so.1
New: /usr/local/lib/libvtkRenderingContextOpenGL2-7.1.so.1
New: /usr/local/lib/libvtkRenderingContext2D-7.1.so.1
New: /usr/local/lib/libvtkRenderingFreeType-7.1.so.1
New: /usr/local/lib/libvtkfreetype-7.1.so.1
New: /usr/local/lib/libvtkRenderingOpenGL2-7.1.so.1
New: /usr/local/lib/libvtkImagingCore-7.1.so.1
New: /usr/local/lib/libvtkRenderingCore-7.1.so.1
New: /usr/local/lib/libvtkCommonColor-7.1.so.1
New: /usr/local/lib/libvtkFiltersGeometry-7.1.so.1
New: /usr/local/lib/libvtkFiltersSources-7.1.so.1
New: /usr/local/lib/libvtkFiltersGeneral-7.1.so.1
New: /usr/local/lib/libvtkCommonComputationalGeometry-7.1.so.1
New: /usr/local/lib/libvtkFiltersCore-7.1.so.1
New: /usr/local/lib/libvtkIOImage-7.1.so.1
New: /usr/local/lib/libvtkCommonExecutionModel-7.1.so.1
New: /usr/local/lib/libvtkCommonDataModel-7.1.so.1
New: /usr/local/lib/libvtkCommonTransforms-7.1.so.1
New: /usr/local/lib/libvtkCommonMisc-7.1.so.1
New: /usr/local/lib/libvtkCommonMath-7.1.so.1
New: /usr/local/lib/libvtkCommonSystem-7.1.so.1
New: /usr/local/lib/libvtkCommonCore-7.1.so.1
New: /usr/local/lib/libvtksys-7.1.so.1
New: /usr/local/lib/libvtkDICOMParser-7.1.so.1
New: /usr/local/lib/libvtkmetaio-7.1.so.1
New: /usr/local/lib/libvtkpng-7.1.so.1
New: /usr/local/lib/libvtktiff-7.1.so.1
New: /usr/local/lib/libvtkzlib-7.1.so.1
New: /usr/local/lib/libvtkjpeg-7.1.so.1
New: /usr/lib/x86_64-linux-gnu/libm.so
New: /usr/lib/x86_64-linux-gnu/libSM.so
New: /usr/lib/x86_64-linux-gnu/libICE.so
New: /usr/lib/x86_64-linux-gnu/libX11.so
New: /usr/lib/x86_64-linux-gnu/libXext.so
New: /usr/lib/x86_64-linux-gnu/libXt.so
New: /usr/local/lib/libvtkglew-7.1.so.1
New: /usr/local/lib/libpcl_ml.so
New: /usr/local/lib/libpcl_filters.so
New: /usr/local/lib/libpcl_sample_consensus.so
New: /usr/local/lib/libpcl_visualization.so
New: /usr/local/lib/libpcl_search.so
New: /usr/local/lib/libpcl_kdtree.so
New: /usr/local/lib/libpcl_io.so
New: /usr/local/lib/libpcl_octree.so
New: /usr/local/lib/libpcl_common.so
New: /usr/lib/x86_64-linux-gnu/libqhull.so
New: CMakeFiles/New.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/alessio/dev/BasketFin/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable New"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/New.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/New.dir/build: New

.PHONY : CMakeFiles/New.dir/build

CMakeFiles/New.dir/requires: CMakeFiles/New.dir/new_planes.cpp.o.requires
CMakeFiles/New.dir/requires: CMakeFiles/New.dir/classes/newBasketModel.cpp.o.requires

.PHONY : CMakeFiles/New.dir/requires

CMakeFiles/New.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/New.dir/cmake_clean.cmake
.PHONY : CMakeFiles/New.dir/clean

CMakeFiles/New.dir/depend:
	cd /home/alessio/dev/BasketFin/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/alessio/dev/BasketFin /home/alessio/dev/BasketFin /home/alessio/dev/BasketFin/build /home/alessio/dev/BasketFin/build /home/alessio/dev/BasketFin/build/CMakeFiles/New.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/New.dir/depend
