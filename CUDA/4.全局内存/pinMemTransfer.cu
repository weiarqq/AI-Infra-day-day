#include "common.h"
#include <cstdlib>
#include <cstring>

int main(int argc, char** argv)
{
    int dev = 0;
    cudaSetDevice(dev);

    unsigned int isize = 1 << 22;
    unsigned int nBytes = isize * sizeof(float);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);

    if (!deviceProp.canMapHostMemory) {
        printf("Device %d does not support mapping CPU host memory!\n", dev);
        CHECK(cudaDeviceReset());
        exit(EXIT_SUCCESS);
    }

    printf("%s starting at ", argv[0]);
    printf("device %d: %s memory size %d nbyte %5.2fMB\n", dev,
        deviceProp.name, isize, nBytes / (1024.0f * 1024.0f));

    float* h_a;
    CHECK(cudaMallocHost((float**)&h_a, nBytes));

    float* d_a;
    CHECK(cudaMalloc((void**)&d_a, nBytes));
    memset(h_a, 0, nBytes);

    for (int i = 0; i < isize; i++) {
        h_a[i] = 0.7f;
    }
    CHECK(cudaMemcpy(d_a, h_a, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(h_a, d_a, nBytes, cudaMemcpyDeviceToHost));

    cudaFree(d_a);
    cudaFreeHost(h_a);

    cudaDeviceReset();
    return 0;
}
