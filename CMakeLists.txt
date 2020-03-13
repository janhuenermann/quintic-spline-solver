cmake_minimum_required(VERSION 3.10)

set (CMAKE_CXX_STANDARD 20)

# set the project name
project(quintic_solver)

# Use OpenCV
find_package( OpenCV REQUIRED )

# Eigen
find_package( Eigen3 REQUIRED )

include_directories(${EIGEN3_INCLUDE_DIR})

# add the executable
add_executable(example main.cpp spline.cpp)

# link target
target_link_libraries(example ${OpenCV_LIBS} ${Eigen3_LIBS})

target_compile_options(example PRIVATE -O3)
