cmake_minimum_required(VERSION 3.18)
project(cuda-day-day VERSION 0.1.0 LANGUAGES C CXX CUDA)

find_package(OpenMP REQUIRED)
find_library(CUBLAS_LIB cublas PATHS $ENV{CUDA_HOME}/lib64 /usr/local/cuda/lib64)
find_library(CURAND_LIB curand PATHS $ENV{CUDA_HOME}/lib64 /usr/local/cuda/lib64)
find_library(CUFFT_LIB cufft PATHS $ENV{CUDA_HOME}/lib64 /usr/local/cuda/lib64)
find_library(CUSPARSE_LIB cusparse PATHS $ENV{CUDA_HOME}/lib64 /usr/local/cuda/lib64)

# 验证库是否找到
if(NOT CUBLAS_LIB)
    message(FATAL_ERROR "cublas library not found! Set CUDA_HOME environment variable.")
endif()
if(NOT CURAND_LIB)
    message(FATAL_ERROR "curand library not found! Set CUDA_HOME environment variable.")
endif()
if(NOT CUFFT_LIB)
    message(FATAL_ERROR "cufft library not found! Set CUDA_HOME environment variable.")
endif()
if(NOT CUSPARSE_LIB)
    message(FATAL_ERROR "cusparse library not found! Set CUDA_HOME environment variable.")
endif()
set(COMMON_FILES common.cpp)

add_executable(cublas cublas.cu)
target_link_libraries(cublas PRIVATE ${CUBLAS_LIB})
set_target_properties(cublas PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

# add_executable(cuda-openacc cuda-openacc.cu)
# target_link_libraries(cuda-openacc PRIVATE ${CUBLAS_LIB} ${CURAND_LIB})
# set_target_properties(cuda-openacc PROPERTIES
#     RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
# )

# add_executable(cufft-multi cufft-multi.cu)
# target_link_libraries(cufft-multi PRIVATE ${CUFFT_LIB})
# set_target_properties(cufft-multi PROPERTIES
#     RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
# )

# add_executable(cufft cufft.cu)
# target_link_libraries(cufft PRIVATE ${CUFFT_LIB})
# set_target_properties(cufft PROPERTIES
#     RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
# )

# add_executable(cusparse cusparse.cu)
# target_link_libraries(cusparse PRIVATE ${CUSPARSE_LIB})
# set_target_properties(cusparse PROPERTIES
#     RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
# )

# add_executable(rand-kernel rand-kernel.cu)
# target_link_libraries(rand-kernel PRIVATE ${CURAND_LIB})
# set_target_properties(rand-kernel PROPERTIES
#     RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
# )

# add_executable(replace-rand-streams replace-rand-streams.cu)
# target_link_libraries(replace-rand-streams PRIVATE ${CURAND_LIB})
# set_target_properties(replace-rand-streams PROPERTIES
#     RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
# )

# add_executable(replace-rand replace-rand.cu)
# target_link_libraries(replace-rand PRIVATE ${CURAND_LIB})
# set_target_properties(replace-rand PROPERTIES
#     RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
# )

# add_executable(simple-data simple-data.c  ${COMMON_FILES})
# add_executable(simple-kernels simple-kernels.c  ${COMMON_FILES})
# add_executable(simple-parallel simple-parallel.c  ${COMMON_FILES})

