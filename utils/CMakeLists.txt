cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
#set (CMAKE_CXX_STANDARD 11)
project(MKA_utils)
find_package(PCL 1.7 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

set (CMAKE_CXX_STANDARD 11)
add_executable (show_traj show_traj.cpp)
target_link_libraries (show_traj ${PCL_LIBRARIES})





