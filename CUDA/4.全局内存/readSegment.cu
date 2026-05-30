#include "common.h"
#include <cuda_runtime.h>

void sumArraysOnHost(float* A, float* B, float* C, const int n, int offset)
{
    for (int idx = 0, k = offset; idx < n; idx++, k++) {
        C[idx] = A[k] + B[k];
    }
}

__global__ void warmup(float* A, float* B, float* C, const int n, int offset)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int k = i + offset;

    if (k < n)
        C[i] = A[k] + B[k];
}

__global__ void readOffset(float* A, float* B, float* C, const int n,
    int offset)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int k = i + offset;

    if (k < n)
        C[i] = A[k] + B[k];
}
int main(int argc, char** argv)
{

    // printf("%s Starting...\n", argv[0]);
    int dev = 0;
    double iStart, iElaps;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    // printf("Use Device: %d %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    int nElem = 1 << 22;
    int nBytes = nElem * sizeof(float);
    printf("Vector size %d nbytes  %3.0f MB\n", nElem,
        (float)nBytes / (1024.0f * 1024.0f));

    int offset = 0;
    int blockSize = 512;
    if (argc > 1)
        offset = atoi(argv[1]);
    if (argc > 2)
        blockSize = atoi(argv[2]);

    dim3 block(blockSize, 1);
    dim3 grid((nElem + block.x - 1) / block.x, 1);

    float *h_A, *h_B, *h_Hs, *h_Ds;
    h_A = (float*)malloc(nBytes);
    h_B = (float*)malloc(nBytes);
    h_Hs = (float*)malloc(nBytes);
    h_Ds = (float*)malloc(nBytes);

    initialData(h_A, nElem);
    memcpy(h_B, h_A, nBytes);

    // memset(h_Hs, 0, nElem);
    // memset(h_Ds, 0, nElem);

    sumArraysOnHost(h_A, h_B, h_Hs, nElem, offset);

    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, nBytes);
    cudaMalloc((void**)&d_B, nBytes);
    cudaMalloc((void**)&d_C, nBytes);

    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_A, nBytes, cudaMemcpyHostToDevice);

    iStart = cpuSecond();
    warmup<<<grid, block>>>(d_A, d_B, d_C, nElem, offset);
    CHECK(cudaDeviceSynchronize());
    iElaps = cpuSecond() - iStart;
    printf("warmup <<<(%d, %d), (%d, %d)>>> elapsed %fsec\n", grid.x, grid.y, block.x, block.y, iElaps);
    CHECK(cudaGetLastError());

    iStart = cpuSecond();
    readOffset<<<grid, block>>>(d_A, d_B, d_C, nElem, offset);
    CHECK(cudaDeviceSynchronize());
    iElaps = cpuSecond() - iStart;
    printf("readOffset <<<(%d, %d), (%d, %d)>>> offset %d elapsed %fsec\n", grid.x, grid.y, block.x, block.y, offset, iElaps);

    CHECK(cudaGetLastError());

    CHECK(cudaMemcpy(h_Ds, d_C, nBytes, cudaMemcpyDeviceToHost));
    checkResult(h_Hs, h_Ds, nElem - offset);

    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));

    free(h_A);
    free(h_B);
    // free(h_Hs);
    // free(h_Ds);

    CHECK(cudaDeviceReset());

    return 0;
}