#include "common.h"
#include <stdio.h>

__device__ void myAtomicAdd(int* address, int incr)
{
    int guess = *address;
    // guess 初始化为0
    // 如果未替换，*address 没变
    int value = atomicCAS(address, guess, guess + incr);
    while (value != guess) {
        guess = value;
        value = atomicCAS(address, guess, guess + incr);
    }
}

__global__ void kernel(int* sharedInteger)
{
    myAtomicAdd(sharedInteger, 1);
}

int main(int argc, char** argv)
{
    int h_sharedInteger;
    int* d_sharedInteger;
    CHECK(cudaMalloc((void**)&d_sharedInteger, sizeof(int)));
    CHECK(cudaMemset(d_sharedInteger, 0x00, sizeof(int)));

    kernel<<<4, 128>>>(d_sharedInteger);

    CHECK(cudaMemcpy(&h_sharedInteger, d_sharedInteger, sizeof(int),
        cudaMemcpyDeviceToHost));
    printf("4 x 128 increments led to value of %d\n", h_sharedInteger);

    return 0;
}
