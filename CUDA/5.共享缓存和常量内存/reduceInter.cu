#include "common.h"
#include <cuda_runtime.h>
#include <stdio.h>

#define DIM 128
int recursiveReduce(int* data, int const size)
{
    if (size == 1)
        return data[0];

    int const stride = size / 2;

    for (int i = 0; i < stride; i++)
        data[i] += data[i + stride];

    return recursiveReduce(data, stride);
}

__global__ void reduceGmem(int* g_idata, int* g_odata, unsigned int n)
{

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 找到当前block处理数组段中的位置
    int* idata = g_idata + blockIdx.x * blockDim.x;

    if (idx >= n)
        return;

    if (blockDim.x >= 1024 && tid < 512)
        idata[tid] += idata[tid + 512];
    __syncthreads();
    if (blockDim.x >= 512 && tid < 256)
        idata[tid] += idata[tid + 256];
    __syncthreads();
    if (blockDim.x >= 256 && tid < 128)
        idata[tid] += idata[tid + 128];
    __syncthreads();
    if (blockDim.x >= 128 && tid < 64)
        idata[tid] += idata[tid + 64];
    __syncthreads();
    if (tid < 32) {
        // volatile修饰符用来确保当线程束在锁步中执行时，只有最新数值能被读取。
        volatile int* vmem = idata; // 每条指令后 线程束隐式同步 必须使用 volatile计算后存储到内存而不是临时放到寄存器
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid + 8];
        vmem[tid] += vmem[tid + 4];
        vmem[tid] += vmem[tid + 2];
        vmem[tid] += vmem[tid + 1];
    }
    // 该数组段，即该block需要处理的数组段 相加的结果 放在idata[0], 然后将结果放到g_odata对应的block.x的位置
    if (tid == 0)
        g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceSmem(int* g_idata, int* g_odata, unsigned int n)
{
    __shared__ int smem[DIM];
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 找到当前block处理数组段中的位置
    int* idata = g_idata + blockIdx.x * blockDim.x;

    if (idx >= n)
        return;
    smem[tid] = idata[tid];
    if (blockDim.x >= 1024 && tid < 512)
        smem[tid] += smem[tid + 512];
    __syncthreads();
    if (blockDim.x >= 512 && tid < 256)
        smem[tid] += smem[tid + 256];
    __syncthreads();
    if (blockDim.x >= 256 && tid < 128)
        smem[tid] += smem[tid + 128];
    __syncthreads();
    if (blockDim.x >= 128 && tid < 64)
        smem[tid] += smem[tid + 64];
    __syncthreads();
    if (tid < 32) {
        // volatile修饰符用来确保当线程束在锁步中执行时，只有最新数值能被读取。
        volatile int* vmem = smem; // 每条指令后 线程束隐式同步 必须使用 volatile计算后存储到内存而不是临时放到寄存器
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid + 8];
        vmem[tid] += vmem[tid + 4];
        vmem[tid] += vmem[tid + 2];
        vmem[tid] += vmem[tid + 1];
    }
    // 该数组段，即该block需要处理的数组段 相加的结果 放在idata[0], 然后将结果放到g_odata对应的block.x的位置
    if (tid == 0)
        g_odata[blockIdx.x] = smem[0];
}

__global__ void reduceSmemUnroll(int* g_idata, int* g_odata, unsigned int n)
{
    __shared__ int smem[DIM];
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 4 + threadIdx.x;

    // 找到当前block处理数组段中的位置
    int tmpSum = 0;
    if (idx + 3 * blockDim.x <= n) {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + blockDim.x * 2];
        int a4 = g_idata[idx + blockDim.x * 3];
        tmpSum = a1 + a2 + a3 + a4;
    }
    smem[tid] = tmpSum;
    __syncthreads();
    if (idx >= n)
        return;
    if (blockDim.x >= 1024 && tid < 512)
        smem[tid] += smem[tid + 512];
    __syncthreads();
    if (blockDim.x >= 512 && tid < 256)
        smem[tid] += smem[tid + 256];
    __syncthreads();
    if (blockDim.x >= 256 && tid < 128)
        smem[tid] += smem[tid + 128];
    __syncthreads();
    if (blockDim.x >= 128 && tid < 64)
        smem[tid] += smem[tid + 64];
    __syncthreads();
    if (tid < 32) {
        // volatile修饰符用来确保当线程束在锁步中执行时，只有最新数值能被读取。
        volatile int* vmem = smem; // 每条指令后 线程束隐式同步 必须使用 volatile计算后存储到内存而不是临时放到寄存器
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid + 8];
        vmem[tid] += vmem[tid + 4];
        vmem[tid] += vmem[tid + 2];
        vmem[tid] += vmem[tid + 1];
    }
    // 该数组段，即该block需要处理的数组段 相加的结果 放在idata[0], 然后将结果放到g_odata对应的block.x的位置
    if (tid == 0)
        g_odata[blockIdx.x] = smem[0];
}

__global__ void reduceSmemUnrollDyn(int* g_idata, int* g_odata, unsigned int n)
{
    extern __shared__ int smem[];
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 4 + threadIdx.x;

    // 找到当前block处理数组段中的位置
    int tmpSum = 0;
    if (idx + 3 * blockDim.x <= n) {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + blockDim.x * 2];
        int a4 = g_idata[idx + blockDim.x * 3];
        tmpSum = a1 + a2 + a3 + a4;
    }
    smem[tid] = tmpSum;
    __syncthreads();
    if (idx >= n)
        return;
    if (blockDim.x >= 1024 && tid < 512)
        smem[tid] += smem[tid + 512];
    __syncthreads();
    if (blockDim.x >= 512 && tid < 256)
        smem[tid] += smem[tid + 256];
    __syncthreads();
    if (blockDim.x >= 256 && tid < 128)
        smem[tid] += smem[tid + 128];
    __syncthreads();
    if (blockDim.x >= 128 && tid < 64)
        smem[tid] += smem[tid + 64];
    __syncthreads();
    if (tid < 32) {
        // volatile修饰符用来确保当线程束在锁步中执行时，只有最新数值能被读取。
        volatile int* vmem = smem; // 每条指令后 线程束隐式同步 必须使用 volatile计算后存储到内存而不是临时放到寄存器
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid + 8];
        vmem[tid] += vmem[tid + 4];
        vmem[tid] += vmem[tid + 2];
        vmem[tid] += vmem[tid + 1];
    }
    // 该数组段，即该block需要处理的数组段 相加的结果 放在idata[0], 然后将结果放到g_odata对应的block.x的位置
    if (tid == 0)
        g_odata[blockIdx.x] = smem[0];
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
    reduceGmem<<<grid, block>>>(g_idata, g_odata, size);
    CHECK(cudaDeviceSynchronize());
    cudaMemcpy(h_odata, g_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost);

    int gpu_sum = 0;
    for (int i = 0; i < grid.x; i++)
        gpu_sum += h_odata[i];

    printf("reduceGmem          : %d <<<grid %d block %d>>>\n", gpu_sum, grid.x,
        block.x);

    CHECK(cudaMemcpy(g_idata, h_idata, nbytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    reduceSmem<<<grid, block>>>(g_idata, g_odata, size);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(h_odata, g_odata, grid.x * sizeof(int),
        cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x; i++)
        gpu_sum += h_odata[i];

    printf("reduceSmem          : %d <<<grid %d block %d>>>\n", gpu_sum, grid.x,
        block.x);

    CHECK(cudaMemcpy(g_idata, h_idata, nbytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    reduceSmemUnroll<<<grid.x / 4, block>>>(g_idata, g_odata, size);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(h_odata, g_odata, grid.x / 4 * sizeof(int),
        cudaMemcpyDeviceToHost));
    gpu_sum = 0;
    for (int i = 0; i < grid.x / 4; i++)
        gpu_sum += h_odata[i];
    printf("reduceSmemUnroll   : %d <<<grid %d block %d>>>\n", gpu_sum,
        grid.x / 4, block.x);

    CHECK(cudaMemcpy(g_idata, h_idata, nbytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    reduceSmemUnrollDyn<<<grid.x / 4, block>>>(g_idata, g_odata, size);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(h_odata, g_odata, grid.x / 4 * sizeof(int),
        cudaMemcpyDeviceToHost));
    gpu_sum = 0;
    for (int i = 0; i < grid.x / 4; i++)
        gpu_sum += h_odata[i];
    printf("reduceSmemUnrollDyn   : %d <<<grid %d block %d>>>\n", gpu_sum,
        grid.x / 4, block.x);

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