#include "common.h"
#include <cuda_runtime.h>
#include <stdio.h>

#define DIM 128
#define SMEMDIM 4 // 128/32 = 8

int recursiveReduce(int* data, int const size)
{
    if (size == 1)
        return data[0];

    int const stride = size / 2;

    for (int i = 0; i < stride; i++)
        data[i] += data[i + stride];

    return recursiveReduce(data, stride);
}

__inline__ __device__ int warpReduce(int localSum)
{
    // 指令同步
    localSum += __shfl_xor_sync(0xffffffff, localSum, 16);
    localSum += __shfl_xor_sync(0xffffffff, localSum, 8);
    localSum += __shfl_xor_sync(0xffffffff, localSum, 4);
    localSum += __shfl_xor_sync(0xffffffff, localSum, 2);
    localSum += __shfl_xor_sync(0xffffffff, localSum, 1);

    return localSum;
}

__global__ void reduceShfl(int* g_idata, int* g_odata, unsigned int n)
{
    // shared memory for each warp sum
    __shared__ int smem[SMEMDIM];

    // boundary check
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n)
        return;

    // calculate lane index and warp index
    int laneIdx = threadIdx.x % warpSize;
    int warpIdx = threadIdx.x / warpSize;

    // blcok-wide warp reduce
    int localSum = warpReduce(g_idata[idx]);

    // save warp sum to shared memory
    if (laneIdx == 0)
        smem[warpIdx] = localSum;

    // block synchronization
    __syncthreads();

    // last warp reduce
    if (threadIdx.x < warpSize)
        localSum = (threadIdx.x < SMEMDIM) ? smem[laneIdx] : 0;

    if (warpIdx == 0)
        localSum = warpReduce(localSum);

    // write result for this block to global mem
    if (threadIdx.x == 0)
        g_odata[blockIdx.x] = localSum;
}

int main(int argc, char** argv)
{

    printf("%s Starting...\n", argv[0]);
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Use Device: %d %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    int size = 1 << 24;
    size_t nbytes = size * sizeof(int);
    int *h_idata, *tmp;
    h_idata = (int*)malloc(size * sizeof(int));
    tmp = (int*)malloc(size * sizeof(int));

    for (int i = 0; i < size; i++) {
        h_idata[i] = (int)(rand() & 0xFF);
    }

    memcpy(tmp, h_idata, size * sizeof(int));
    int cpu_sum = recursiveReduce(tmp, size);
    printf("cpu reduce          : %d\n", cpu_sum);

    int block_size = DIM;
    dim3 block(block_size, 1);
    dim3 grid((size + block.x - 1) / block.x, 1);

    int *g_idata, *g_odata;
    cudaMalloc((void**)&g_idata, size * sizeof(int));
    cudaMalloc((void**)&g_odata, grid.x * sizeof(int));

    int* h_odata = (int*)malloc(grid.x * sizeof(int));

    CHECK(cudaMemcpy(g_idata, h_idata, nbytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    reduceShfl<<<grid.x, block>>>(g_idata, g_odata, size);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(h_odata, g_odata, grid.x * sizeof(int),
        cudaMemcpyDeviceToHost));
    int gpu_sum = 0;
    for (int i = 0; i < grid.x; i++)
        gpu_sum += h_odata[i];
    printf("reduceShfl   : %d <<<grid %d block %d>>>\n", gpu_sum,
        grid.x, block.x);

    // free host memory
    free(h_idata);
    free(h_odata);

    // free device memory
    CHECK(cudaFree(g_idata));
    CHECK(cudaFree(g_odata));

    // reset device
    CHECK(cudaDeviceReset());
    return 0;
}