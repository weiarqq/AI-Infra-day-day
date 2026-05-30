#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>


/*
用CPU计时器计时
*/

#ifndef MY_MACROS_H // 头文件保护（防止重复包含）
#define MY_MACROS_H

#define CHECK(call)                                                             \
    {                                                                           \
        const cudaError_t error = call;                                         \
        if (error != cudaSuccess) {                                             \
            printf("Error: %s:%d", __FILE__, __LINE__);                         \
            printf("code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
            exit(1);                                                            \
        }                                                                       \
    }
#endif

void checkResult(float* hostRef, float* gpuRef, const int N)
{
    double epsilon = 1.0E-8;
    int match = 1;
    for (int i = 0; i < N; i++) {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon) {
            match = 0;
            printf("Arrays do not match!\n");
            printf("host %5.2f gpu %5.2f at current %d\n",
                hostRef[i], gpuRef[i], i);
            break;
        }
    }
    if (match)
        printf("Arrays match.\n\n");
    return;
}

double cpuSecond()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);

    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

void sumMatrixOnHost(float* A, float* B, float* C, const int nx, const int ny)
{
    float *ia = A;
    float *ib = B;
    float *ic = C;
    for(int iy=0; iy < ny; iy ++){
        for (int ix = 0; ix < nx; ix++) {
            ic[ix] = ia[ix] + ib[ix];
        }
        ia += nx;
        ib += nx;
        ic += nx;
    }
}

__global__ void sumMatrixOnGPU1D(float* A, float* B, float* C, const int nx, const int ny)
{
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
    if(ix <nx){
        for(int iy = 0; iy < ny; iy++){
            // 每个线程处理 当前行的第ix个元素， 全局内存索引位置：第iy行 (iy*nx)的 第ix个
            unsigned int idx = ix + iy*nx;
            C[idx] = A[idx] + B[idx];
        }
    }
}

void initialData(float* ip, int size)
{
    time_t t;
    srand((unsigned int)time(&t));
    for (int i = 0; i < size; i++) {
        ip[i] = static_cast<float>((rand() & 0xFF) / 10.0f);
    }
}

int main(int argc, char** argv)
{
    printf("%s Starting...\n", argv[0]);
    int dev = 0;
    double iStart, iElaps;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Use Device: %d %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    int nx = 1 << 14;
    int ny = 1 << 14;
    int nElem = nx * ny;
    printf("Vector size %d\n", nElem);
    size_t nBytes = nElem * sizeof(float);

    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A = (float*)malloc(nBytes);
    h_B = (float*)malloc(nBytes);
    hostRef = (float*)malloc(nBytes);
    gpuRef = (float*)malloc(nBytes);

    iStart = cpuSecond();
    initialData(h_A, nElem);
    initialData(h_B, nElem);
    iElaps = cpuSecond() - iStart;

    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    iStart = cpuSecond();
    sumMatrixOnHost(h_A, h_B, hostRef, nx, ny);
    iElaps = cpuSecond() - iStart;
    printf("sumArraysOnHost Time elaspsed %f sec \n", iElaps);

    float *d_A, *d_B, *d_C;
    // void** 类型 指针的指针
    cudaMalloc((void**)&d_A, nBytes);
    cudaMalloc((void**)&d_B, nBytes);
    cudaMalloc((void**)&d_C, nBytes);

    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);

    int dimx = 32;
    int dimy = 1;
    dim3 block(dimx, dimy);
    // 按照nx来划分线程，则核函数中，则每个线程需要计算ny个元素
    dim3 grid((nx + block.x - 1) / block.x);

    iStart = cpuSecond();
    sumMatrixOnGPU1D<<<grid, block>>>(d_A, d_B, d_C, nx, ny);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    printf("sumArraysOnDevice <<<%d, %d>>> Time elaspsed %f sec \n", grid.x, block.x, iElaps);
    cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);

    checkResult(hostRef, gpuRef, nElem);

    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaDeviceReset();
    return 0;
}