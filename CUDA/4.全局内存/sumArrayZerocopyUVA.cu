#include "common.h"
#include <cstdlib>

void sumArraysOnHost(float* A, float* B, float* C, const int N)
{
    for (int idx = 0; idx < N; idx++) {
        C[idx] = A[idx] + B[idx];
    }
}

__global__ void sumArrays(float* A, float* B, float* C, const int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N)
        C[i] = A[i] + B[i];
}

__global__ void sumArraysZeroCopy(float* A, float* B, float* C, const int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N)
        C[i] = A[i] + B[i];
}

int main(int argc, char** argv)
{
    int dev = 0;
    CHECK(cudaSetDevice(dev));

    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));

    if (!deviceProp.canMapHostMemory) {
        printf("Device %d does not support mapping CPU host memory!\n", dev);
        CHECK(cudaDeviceReset());
        exit(EXIT_SUCCESS);
    }

    printf("Using Device %d: %s ", dev, deviceProp.name);

    int ipower = 10;

    if (argc > 1)
        ipower = atoi(argv[1]);

    int nElem = 1 << ipower;
    size_t nBytes = nElem * sizeof(float);

    if (ipower < 18) {
        printf("Vector size %d power %d  nbytes  %3.0f KB\n", nElem, ipower,
            (float)nBytes / (1024.0f));
    } else {
        printf("Vector size %d power %d  nbytes  %3.0f MB\n", nElem, ipower,
            (float)nBytes / (1024.0f * 1024.0f));
    }

    float *h_A, *h_B, *h_C, *gpuRef;
    h_A = (float*)malloc(nBytes);
    h_B = (float*)malloc(nBytes);
    h_C = (float*)malloc(nBytes);
    gpuRef = (float*)malloc(nBytes);

    initialData(h_A, nElem);
    initialData(h_B, nElem);
    memset(h_C, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    sumArraysOnHost(h_A, h_B, h_C, nElem);

    float *d_A, *d_B, *d_C;
    CHECK(cudaMalloc((float**)&d_A, nBytes));
    CHECK(cudaMalloc((float**)&d_B, nBytes));
    CHECK(cudaMalloc((float**)&d_C, nBytes));

    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));

    int iLen = 512;
    dim3 block(iLen);
    dim3 grid((nElem + block.x - 1) / block.x);

    sumArrays<<<grid, block>>>(d_A, d_B, d_C, nElem);

    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    checkResult(h_C, gpuRef, nElem);

    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));

    free(h_A);
    free(h_B);

    CHECK(cudaHostAlloc((void**)&h_A, nBytes, cudaHostAllocMapped));
    CHECK(cudaHostAlloc((void**)&h_B, nBytes, cudaHostAllocMapped));

    initialData(h_A, nElem);
    initialData(h_B, nElem);
    memset(h_C, 0, nBytes);
    memset(gpuRef, 0, nBytes);
    // CHECK(cudaHostGetDevicePointer((void**)d_A, (void*)h_A, 0))
    // CHECK(cudaHostGetDevicePointer((void**)d_B, (void*)h_B, 0))
    // sumArraysOnHost(h_A, h_B, h_C, nElem);

    sumArraysZeroCopy<<<grid, block>>>(h_A, h_B, d_C, nElem);

    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    checkResult(h_C, gpuRef, nElem);

    CHECK(cudaFree(d_C));
    CHECK(cudaFreeHost(h_A));
    CHECK(cudaFreeHost(h_B));

    free(h_C);
    free(gpuRef);

    CHECK(cudaDeviceReset());
    return EXIT_SUCCESS;
}
