cmake_minimum_required(VERSION 2.8)
project(Hough)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(Hough main.cpp)
target_link_libraries(Hough ${OpenCV_LIBS})
