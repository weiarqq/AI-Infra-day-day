#include "common.h"
#include <cuda_runtime.h>
#include <stdio.h>
__global__ void nestedHelloWorld(const int size, int iDepth){
    int tid = threadIdx.x;
    printf("Recursion=%d: Hello World from thread %d block %d\n", iDepth, tid,
           blockIdx.x);
    if(size == 1) return;
    int nthreads = size >>1;
    if(tid == 0 && nthreads > 0){
        nestedHelloWorld<<<1,nthreads>>>(nthreads, ++iDepth);
        printf("-------> nested execution depth: %d\n", iDepth);
    }
}

int main(int argc, char** argv){
    int size = 8;
    int blocksize = 8;   // initial block size
    int igrid = 1;

    if(argc > 1)
    {
        igrid = atoi(argv[1]);
        size = igrid * blocksize;
    }

    dim3 block(blocksize);
    dim3 grid((size+block.x-1)/ block.x);
    nestedHelloWorld<<<grid, block>>>(size, 1);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceReset());
    return 0;
}