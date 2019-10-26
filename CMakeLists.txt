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


add_executable (Planes planes.cpp classes/basketModel.cpp) 
target_link_libraries (Planes ${PCL_LIBRARIES})
target_link_libraries(Planes ${Boost_LIBRARIES})

add_executable (New new_planes.cpp classes/newBasketModel.cpp) 
target_link_libraries (New ${PCL_LIBRARIES})
target_link_libraries(New ${Boost_LIBRARIES})

add_executable (Test test_cylinder.cpp) 
target_link_libraries (Test ${PCL_LIBRARIES})
target_link_libraries(Test ${Boost_LIBRARIES})


add_executable (Sequence sequence.cpp classes/findTarget.cpp classes/basketModel.cpp) 
target_link_libraries (Sequence ${PCL_LIBRARIES})
target_link_libraries(Sequence ${Boost_LIBRARIES})

# basket5 center and axis dir hardcoded
add_executable (SequenceB5 sequenceB5.cpp classes/findTarget.cpp classes/newBasketModel.cpp) 
target_link_libraries (SequenceB5 ${PCL_LIBRARIES})
target_link_libraries(SequenceB5 ${Boost_LIBRARIES})


add_executable (Pico pico.cpp) 
target_link_libraries (Pico ${PCL_LIBRARIES})
target_link_libraries(Pico ${Boost_LIBRARIES})

add_executable (FinModel fin_model.cpp) 
target_link_libraries (FinModel ${PCL_LIBRARIES})
target_link_libraries(FinModel ${Boost_LIBRARIES})




























