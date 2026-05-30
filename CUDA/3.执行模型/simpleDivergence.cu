#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "common.h"


__global__ void mathKernel1(float *c){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    float a, b;
    a = b = 0.0f;
    if(tid % 2 ==0){
        a = 100.0f;
    }else{
        b = 200.0f;
    }
    c[tid] = a + b;
}

// 使用warpSize 来区分同一线程束的线程, 避免线程束分化
__global__ void mathKernel2(float *c){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    float a, b;
    a = b = 0.0f;
    if((tid/warpSize) % 2 ==0){
        a = 100.0f;
    }else{
        b = 200.0f;
    }
    c[tid] = a + b;
}

__global__ void mathKernel3(float *c){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    float a, b;
    a = b = 0.0f;
    bool ipred = (tid % 2 == 0);
    if(ipred){
        a = 100.0f;
    }
    if(!ipred){
        b = 200.0f;
    }
    c[tid] = a + b;
}


int main(int argc, char** argv)
{
    printf("%s Starting...\n", argv[0]);
    int dev = 0;
    double iStart, iElaps;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("%s using Device: %d %s\n", argv[0], dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    int size = 64;
    int blocksize = 64;
    if(argc > 1) size = atoi(argv[1]);
    if(argc > 2) blocksize = atoi(argv[2]);
    printf("Vector size %d\n", size);
    size_t nBytes = size * sizeof(float);

    dim3 block(blocksize, 1);
    dim3 grid((size+block.x -1)/ block.x, 1);
    printf("Execution Configure (block %d grid %d) \n", block.x, grid.x);

    float *d_C;
    cudaMalloc((float**)&d_C, nBytes);
    cudaDeviceSynchronize();

    iStart = cpuSecond();
    mathKernel1<<<grid, block>>>(d_C);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    printf("warmingup <<<%4d %4d>>> elapsed %f sec \n", grid.x, block.x,iElaps);


    iStart = cpuSecond();
    mathKernel1<<<grid, block>>>(d_C);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    printf("mathKernel1 <<<%4d %4d>>> elapsed %f sec \n", grid.x, block.x,iElaps);


    iStart = cpuSecond();
    mathKernel2<<<grid, block>>>(d_C);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    printf("mathKernel2 <<<%4d %4d>>> elapsed %f sec \n", grid.x, block.x,iElaps);


    iStart = cpuSecond();
    mathKernel3<<<grid, block>>>(d_C);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    printf("mathKernel3 <<<%4d %4d>>> elapsed %f sec \n", grid.x, block.x, iElaps);

    cudaFree(d_C);

    cudaDeviceReset();
    return 0;
}