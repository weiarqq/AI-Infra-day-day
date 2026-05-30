cmake_minimum_required(VERSION 3.18)
project(cuda-day-day VERSION 0.1.0 LANGUAGES C CXX CUDA)


# 查询 cudatoolkit
find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)

if (CMAKE_CUDA_COMPILER)
    message(STATUS "nvcc path : ${CMAKE_CUDA_COMPILER}")
else ()
    message(WARNING "nvcc not found. Please check CUDA is installed correctly!")
endif ()

# 指定C++标准（CUDA也继承此标准）
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)  # 显式指定CUDA标准
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA架构（推荐用target属性，而非全局变量）
set(CMAKE_CUDA_ARCHITECTURES 89)

function(add_cudadevrt target_name)
    # 使用${target_name}引用函数参数（变量），而非字面量

    # 启用分离编译（替代手动设置 -rdc=true，CMake会自动处理设备链接）
    set_target_properties(${target_name} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON  # 关键：启用分离编译，自动添加-rdc=true和设备链接步骤
        CUDA_RESOLVE_DEVICE_SYMBOLS ON # 确保设备符号解析（分离编译必需）
    )

    # 链接CUDA库
    target_include_directories(${target_name} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_link_directories(${target_name} PRIVATE ${CUDAToolkit_LIBRARY_DIR})
    # 直接指定库名称（CUDAToolkit提供的变量）
    target_link_libraries(${target_name} PRIVATE
        ${CUDAToolkit_CUDA_LIBRARY}    # 替代CUDAToolkit::CUDA
        ${CUDAToolkit_cudadevrt_LIBRARY} # 替代CUDAToolkit::cudadevrt
    )

endfunction()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler ${OpenMP_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

# 调试/发布模式的CUDA编译选项
option(ENABLE_CUDA_DEBUG "Enable CUDA debug (cuda-gdb)" OFF)
if(ENABLE_CUDA_DEBUG)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")        # enable cuda-gdb (may significantly affect performance on some targets)
else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo") # add line information to all builds for debug tools (exclusive to -G option)
    # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo -Xptxas -dlcm=cg") # add line information to all builds for debug tools (exclusive to -G option)
    message("CUDA compiler:" ${CMAKE_CUDA_FLAGS})
    message("CPP compiler:" ${CMAKE_CXX_FLAGS})

endif()


set(COMMON_FILES common.cpp)

add_executable(simple2DFD simple2DFD.cu  ${COMMON_FILES})
add_executable(simpleC2C.c simpleC2C  ${COMMON_FILES})
add_executable(simpleMultiGPU simpleMultiGPU.cu  ${COMMON_FILES})
add_executable(simpleP2P simpleP2P.c  ${COMMON_FILES})
add_executable(simpleP2P_CUDA_Aware simpleP2P_CUDA_Aware.c  ${COMMON_FILES})
add_executable(simpleP2P_PingPong simpleP2P_PingPong.cu  ${COMMON_FILES})