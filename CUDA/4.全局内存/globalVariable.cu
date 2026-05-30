#include "common.h"

__device__ float devData;

__global__ void checkGlobalVariable()
{
    printf("Device: the value of the global variable is %f\n", devData);

    devData += 2.0f;
}

int main()
{
    float value = 0.0f;
    cudaMemcpyToSymbol(devData, &value, sizeof(float));
    printf("Host:   copied %f to the global variable\n", value);

    checkGlobalVariable<<<1, 1>>>();
    cudaMemcpyFromSymbol(&value, devData, sizeof(float));

    printf("Host:   copied %f to the global variable\n", value);
    cudaDeviceReset();
    return 0;
}