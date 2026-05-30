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
    printf("%s starting at ", argv[0]);
    printf("device %d: %s memory size %d nbyte %5.2fMB\n", dev,
        deviceProp.name, isize, nBytes / (1024.0f * 1024.0f));

    float* h_a;
    h_a = (float*)std::malloc(nBytes);

    float* d_a;
    CHECK(cudaMalloc((void**)&d_a, nBytes));

    for (int i = 0; i < isize; i++) {
        h_a[i] = 0.7f;
    }
    CHECK(cudaMemcpy(d_a, h_a, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(h_a, d_a, nBytes, cudaMemcpyDeviceToHost));

    cudaFree(d_a);
    free(h_a);

    cudaDeviceReset();
    return 0;
}
