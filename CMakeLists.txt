cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

set (CMAKE_CXX_STANDARD 11)


project(Entropy_Filter)

find_package(Boost REQUIRED)

find_package(PCL 1.9 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})


add_executable (FindFin find_fin.cpp classes/findTarget.cpp) #Alignment Prerejective + ICP
target_link_libraries (FindFin ${PCL_LIBRARIES})
target_link_libraries(FindFin ${Boost_LIBRARIES})


add_executable (Pico pico.cpp) 
target_link_libraries (Pico ${PCL_LIBRARIES})
target_link_libraries(Pico ${Boost_LIBRARIES})

add_executable (FinModel fin_model.cpp) 
target_link_libraries (FinModel ${PCL_LIBRARIES})
target_link_libraries(FinModel ${Boost_LIBRARIES})

add_executable (Planes planes.cpp) 
target_link_libraries (Planes ${PCL_LIBRARIES})
target_link_libraries(Planes ${Boost_LIBRARIES})

add_executable (Sequence sequence.cpp) 
target_link_libraries (Sequence ${PCL_LIBRARIES})
target_link_libraries(Sequence ${Boost_LIBRARIES})

































