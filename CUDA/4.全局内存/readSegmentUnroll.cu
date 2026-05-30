#include "common.h"

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

__global__ void readOffsetUnroll2(float* A, float* B, float* C, const unsigned int n,
    int offset)
{
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    unsigned int k = i + offset;

    if (k < n) {
        C[i] = A[k] + B[k];
    }
    if ((k + blockDim.x) < n) {
        C[i + blockDim.x] = A[k + blockDim.x] + B[k + blockDim.x];
    }
}

__global__ void readOffsetUnroll4(float* A, float* B, float* C, const unsigned int n,
    int offset)
{
    unsigned int i = blockIdx.x * blockDim.x * 4 + threadIdx.x;
    unsigned int k = i + offset;

    if (k < n) {
        C[i] = A[k] + B[k];
    }
    if ((k + blockDim.x) < n) {
        C[i + blockDim.x] = A[k + blockDim.x] + B[k + blockDim.x];
    }
    if ((k + 2 * blockDim.x) < n) {
        C[i + blockDim.x * 2] = A[k + blockDim.x * 2] + B[k + blockDim.x * 2];
    }
    if ((k + 3 * blockDim.x) < n) {
        C[i + blockDim.x * 3] = A[k + blockDim.x * 3] + B[k + blockDim.x * 3];
    }
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

    int nElem = 1 << 24;
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

    //  kernel 1:
    iStart = cpuSecond();
    warmup<<<grid, block>>>(d_A, d_B, d_C, nElem, offset);
    CHECK(cudaDeviceSynchronize());
    iElaps = cpuSecond() - iStart;
    printf("warmup     <<< %4d, %4d >>> offset %4d elapsed %f sec\n", grid.x,
        block.x, offset, iElaps);
    CHECK(cudaGetLastError());
    CHECK(cudaMemset(d_C, 0x00, nBytes));

    // kernel 1
    iStart = cpuSecond();
    readOffset<<<grid, block>>>(d_A, d_B, d_C, nElem, offset);
    CHECK(cudaDeviceSynchronize());
    iElaps = cpuSecond() - iStart;
    printf("readOffset <<< %4d, %4d >>> offset %4d elapsed %f sec\n", grid.x,
        block.x, offset, iElaps);
    CHECK(cudaGetLastError());

    // copy kernel result back to host side and check device results
    CHECK(cudaMemcpy(h_Ds, d_C, nBytes, cudaMemcpyDeviceToHost));
    checkResult(h_Hs, h_Ds, nElem - offset);
    CHECK(cudaMemset(d_C, 0x00, nBytes));

    // kernel 2
    iStart = cpuSecond();
    readOffsetUnroll2<<<grid.x / 2, block>>>(d_A, d_B, d_C, nElem, offset);
    CHECK(cudaDeviceSynchronize());
    iElaps = cpuSecond() - iStart;
    printf("unroll2    <<< %4d, %4d >>> offset %4d elapsed %f sec\n",
        grid.x / 2, block.x, offset, iElaps);
    CHECK(cudaGetLastError());

    // copy kernel result back to host side and check device results
    CHECK(cudaMemcpy(h_Ds, d_C, nBytes, cudaMemcpyDeviceToHost));
    checkResult(h_Hs, h_Ds, nElem - offset);
    CHECK(cudaMemset(d_C, 0x00, nBytes));

    // kernel 3
    iStart = cpuSecond();
    readOffsetUnroll4<<<grid.x / 4, block>>>(d_A, d_B, d_C, nElem, offset);
    CHECK(cudaDeviceSynchronize());
    iElaps = cpuSecond() - iStart;
    printf("unroll4    <<< %4d, %4d >>> offset %4d elapsed %f sec\n",
        grid.x / 4, block.x, offset, iElaps);
    CHECK(cudaGetLastError());

    // copy kernel result back to host side and check device results
    CHECK(cudaMemcpy(h_Ds, d_C, nBytes, cudaMemcpyDeviceToHost));
    checkResult(h_Hs, h_Ds, nElem - offset);
    CHECK(cudaMemset(d_C, 0x00, nBytes));

    // free host and device memory
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);

    // reset device
    CHECK(cudaDeviceReset());
    return EXIT_SUCCESS;
}