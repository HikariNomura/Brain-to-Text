cmake_minimum_required(VERSION 3.15)
project(mtrf_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_executable(mtrf
    src/main.cpp
    src/npy.cpp
)

target_include_directories(mtrf PRIVATE
    third_party/eigen
)

if(MSVC)
    target_compile_options(mtrf PRIVATE /O2 /W4)
else()
    target_compile_options(mtrf PRIVATE -O3 -Wall -Wextra)
endif()
