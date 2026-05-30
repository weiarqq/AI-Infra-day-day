#include "common.h"
// #include <__clang_cuda_builtin_vars.h>

#define BDIMX 8
#define BDIMY 2
#define IPAD 2
__global__ void setRowReadRow(int* out)
{
    __shared__ int tile[BDIMY][BDIMX]; // 16,32 ->block(32, 16)-> 16*32
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

    tile[threadIdx.y][threadIdx.x] = idx; // 共享内存的存储操作

    __syncthreads();
    out[idx] = tile[threadIdx.y][threadIdx.x]; // 全局内存的存储操作 共享内存的加载操作
}

__global__ void setColReadCol(int* out)
{
    __shared__ int tile[BDIMX][BDIMY]; // 32, 16
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

    tile[threadIdx.x][threadIdx.y] = idx; // 共享内存的存储操作

    __syncthreads();
    out[idx] = tile[threadIdx.x][threadIdx.y]; // 全局内存的存储操作 共享内存的加载操作
}

__global__ void setRowReadCol(int* out)
{
    __shared__ int tile[BDIMY][BDIMX];
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int row_id = idx / blockDim.y;
    unsigned int col_id = idx % blockDim.y;

    tile[threadIdx.y][threadIdx.x] = idx; // 共享内存的存储操作

    __syncthreads();
    out[idx] = tile[col_id][row_id]; // tile相当于被转置了
}

__global__ void setRowReadColDynamic(int* out)
{
    extern __shared__ int tile[]; // 动态共享内存 必须是一维的 有extern修饰
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

    unsigned int row_id = idx / blockDim.y;
    unsigned int col_id = idx % blockDim.y;
    unsigned int col_idx = col_id * blockDim.x + row_id;

    tile[idx] = idx;

    __syncthreads();
    out[idx] = tile[col_idx]; // 全局内存的存储操作 共享内存的加载操作
}

__global__ void setRowReadColPad(int* out)
{
    __shared__ int tile[BDIMY][BDIMX + IPAD];
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int row_id = idx / blockDim.y;
    unsigned int col_id = idx % blockDim.y;

    tile[threadIdx.y][threadIdx.x] = idx; // 共享内存的存储操作

    __syncthreads();
    out[idx] = tile[col_id][row_id]; // tile相当于被转置了
}
__global__ void setRowReadColDynPad(int* out)
{
    extern __shared__ int tile[];
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int row_id = idx / blockDim.y;
    unsigned int col_id = idx % blockDim.y;

    unsigned int r_idx = threadIdx.y * (blockDim.x + IPAD) + threadIdx.x;
    unsigned int col_idx = col_id * (blockDim.x + IPAD) + row_id;
    tile[r_idx] = idx;

    __syncthreads();
    out[idx] = tile[col_idx]; // 全局内存的存储操作 共享内存的加载操作
}

int main(int argc, char** argv)
{
    printf("%s Starting...\n", argv[0]);
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Use Device: %d %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    dim3 block(BDIMX, BDIMY);
    dim3 grid(1, 1);
    size_t size = BDIMX * BDIMY;
    size_t nBytes = size * sizeof(int);

    int* gpuRes = new int[size];
    int* d_O;
    cudaMalloc((int**)&d_O, nBytes);

    CHECK(cudaMemset(d_O, 0, size * sizeof(int)));
    setColReadCol<<<grid, block>>>(d_O);
    CHECK(cudaMemcpy(gpuRes, d_O, nBytes, cudaMemcpyDeviceToHost));

    printData("set col read col   ", gpuRes, size);

    CHECK(cudaMemset(d_O, 0, size * sizeof(int)));
    setRowReadRow<<<grid, block>>>(d_O);
    CHECK(cudaMemcpy(gpuRes, d_O, nBytes, cudaMemcpyDeviceToHost));

    printData("set row read row   ", gpuRes, size);

    CHECK(cudaMemset(d_O, 0, size * sizeof(int)));
    setRowReadCol<<<grid, block>>>(d_O);
    CHECK(cudaMemcpy(gpuRes, d_O, nBytes, cudaMemcpyDeviceToHost));

    printData("set row read Col   ", gpuRes, size);

    CHECK(cudaMemset(d_O, 0, size * sizeof(int)));
    setRowReadColDynamic<<<grid, block, BDIMX * BDIMY * sizeof(int)>>>(d_O);
    CHECK(cudaMemcpy(gpuRes, d_O, nBytes, cudaMemcpyDeviceToHost));

    printData("set row read Col dynamic   ", gpuRes, size);

    CHECK(cudaMemset(d_O, 0, size * sizeof(int)));
    setRowReadColPad<<<grid, block>>>(d_O);
    CHECK(cudaMemcpy(gpuRes, d_O, nBytes, cudaMemcpyDeviceToHost));

    printData("set row read Col pad   ", gpuRes, size);

    CHECK(cudaMemset(d_O, 0, size * sizeof(int)));
    setRowReadColDynPad<<<grid, block, (BDIMX + 1) * BDIMY * sizeof(int)>>>(d_O);
    CHECK(cudaMemcpy(gpuRes, d_O, nBytes, cudaMemcpyDeviceToHost));

    printData("set row read Col pad   ", gpuRes, size);

    CHECK(cudaFree(d_O));
    free(gpuRes);
    cudaDeviceReset();
    return 0;
}