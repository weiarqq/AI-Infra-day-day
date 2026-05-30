#include <cuda_runtime.h>
#include <stdio.h>

/*
检查块和线程索引
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


void initialInt(int *ip, int size){
    for(int i=0; i<size; i++){
        ip[i] = i;
    }
}


void printMatrix(int *c, int nx, int ny){
    int *ic = c;
    printf("Matrix dim: %d, %d", nx, ny);
    for (int y=0; y < ny; y++){
        for (int x=0; x < nx; x++){
            printf("%3d ", ic[x]);
        }
        ic+=nx; // 指针偏移
        printf("\n");
    }
    printf("\n");

}

__global__ void printThreadIndex(int *A, const int nx, const int ny){
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    int idx = ix + iy * nx;
    printf("threadIdx (%d, %d) blockIdx (%d, %d) map_index (%d, %d) gloabl_index: %d val:%d",
        threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, ix, iy, idx, A[idx]);
}


int main(){
    int nx = 8;
    int ny = 6;
    int device = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, device));
    CHECK(cudaSetDevice(device));

    int nElem = nx * ny;
    int nBytes = nx*ny*sizeof(int);
    int* h_MA;
    h_MA = (int*) malloc(nBytes);

    initialInt(h_MA, nElem);

    printMatrix(h_MA, nx, ny);

    int *d_MA;
    cudaMalloc((void**) &d_MA, nBytes);

    cudaMemcpy(d_MA, h_MA, nBytes, cudaMemcpyHostToDevice);

    dim3 block(4, 2);
    dim3 grid((nx+block.x -1)/block.x, (ny+block.y -1)/block.y);
    printThreadIndex<<<grid, block>>>(d_MA, nx, ny);

    cudaDeviceSynchronize();

    cudaFree(d_MA);
    free(h_MA);

    cudaDeviceReset();
    return 0;
}
