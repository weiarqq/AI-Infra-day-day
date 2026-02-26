#include <iostream>
#include <vector>

// CUTLASS 核心头文件
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/gemm.h"
#include "cutlass/util/tensor_view_io.h"

// 定义矩阵维度
int M = 512;
int N = 512;
int K = 512;

using ElementAccumulator = float;
using ElementComputeEpilogue = float;

using ElementInputA = cutlass::half_t;
using ElementInputB = cutlass::half_t;
using ElementOuputC = cutlass::half_t;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::RowMajor;
using LayoutC = cutlass::layout::RowMajor;

using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutA,
    ElementInputB, LayoutB,
    ElementOutputC, LayoutC,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80>;

int main()
{
    const int M, N, K = 512, 512, 512;
    int lda, ldb, ldc = K, N, N;

    ElementInputA* d_A;
    ElementInputB* d_B;
    ElementOutputC* d_C;

    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(ElementInputA)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(ElementInputB)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(ElementOutputC)));
    float alpha = 1.0f;
    float beta = 0.0f;
    Gemm::Arguments args
    {
        { M, N, K },
            { d_A, lda },
            { d_B, ldb },
            { d_C, ldc },
            { d_C, ldc },
        {
            alpha, beta
        }
    }
    Gemm gemm_op;
    cutlass::Status status = gemm_op.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "CUTLASS cannot implement this GEMM kernel." << std::endl;
        return -1;
    }
    status = gemm_op(args);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "CUTLASS GEMM execution failed." << std::endl;
        return -1;
    }

    std::cout << "GEMM 成功运行 (手动指定 lda/ldb/ldc)" << std::endl;

    // 释放
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
