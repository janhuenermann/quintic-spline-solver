cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_COMPILER "g++")
set(CMAKE_CXX_STANDARD 20)
set(INSTALL_DIR ${CMAKE_SOURCE_DIR}/bin)

# set the project name
project(spline_solver)

# Use OpenCV and Eigen
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(includes/ ${EIGEN3_INCLUDE_DIR} ${OpenCV_INCLUDE_DIR})

# add lib
add_library(spline_solver STATIC src/spline.cpp)

# add the executable
add_executable(example main.cpp)
add_dependencies(example spline_solver)

# link target
target_link_libraries(example spline_solver ${OpenCV_LIBS} ${Eigen3_LIBS})

# turn optimization on
target_compile_options(spline_solver PRIVATE -O3)

# mkdir
file(MAKE_DIRECTORY ${INSTALL_DIR})
install(TARGETS spline_solver DESTINATION ${INSTALL_DIR})