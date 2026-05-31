cmake_minimum_required(VERSION 3.20)

project(sycl-day-day LANGUAGES CXX)

if (NOT CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM")
    message(FATAL_ERROR "This project requires Intel oneAPI DPC++ compiler: icpx")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(day0-test day0/main.cpp)
add_executable(day1-test day1/main.cpp)

target_compile_options(day0-test PRIVATE -fsycl)
target_link_options(day0-test PRIVATE -fsycl)

target_compile_options(day1-test PRIVATE -fsycl)
target_link_options(day1-test PRIVATE -fsycl)