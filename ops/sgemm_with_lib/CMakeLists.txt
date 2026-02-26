cmake_minimum_required(VERSION 3.18)
project(CutlassGemmExample LANGUAGES CXX CUDA)

# 设置 CUDA 架构（根据你的显卡调整，如 80 对应 A100, 86 对应 RTX 30系列, 89 对应 RTX 40系列）
set(CMAKE_CUDA_ARCHITECTURES 80)

# 查找 CUTLASS 头文件路径（请修改为你本地的实际路径）
set(CUTLASS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cutlass/include")

add_executable(cutlass_gemm sgemm_with_cutlass.cu)

target_include_directories(cutlass_gemm PRIVATE 
    ${CUTLASS_DIR}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

set_target_properties(cutlass_gemm PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CXX_STANDARD 17
)