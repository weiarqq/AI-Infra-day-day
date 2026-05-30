#include <stdio.h>

__global__ void helloFromGPU(void)
{
    printf("Hello world from GPU \n");
}

int main()
{

    helloFromGPU<<<1, 10>>>();
    cudaDeviceReset();
}