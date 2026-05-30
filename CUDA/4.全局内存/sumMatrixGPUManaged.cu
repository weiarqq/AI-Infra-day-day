#include "common.h"
#include <cuda_runtime.h>

void sumMatrixOnHost(float* A, float* B, float* C, const int nx, const int ny)
{
    for (int iy = 0; iy < ny; iy++) {
        for (int ix = 0; ix < nx; ix++) {
            C[ix + iy * nx] = A[ix + iy * nx] + B[ix + iy * nx];
        }
    }
}

__global__ void sumMatrix(float* A, float* B, float* C, int nx, int ny)
{
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int idx = iy * nx + ix;
    if (ix < nx && iy < ny) {
        C[idx] = A[idx] + B[idx];
    }
}

int main(int argc, char** argv)
{

    printf("%s Starting...\n", argv[0]);
    int dev = 0;
    double iStart, iElaps;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Use Device: %d %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));
    int nx = 1 << 14;
    int ny = 1 << 14;
    int ishift = 12;
    if (argc > 1)
        ishift = atoi(argv[1]);
    nx = ny = 1 << ishift;
    printf("Matrix size: nx %d ny %d\n", nx, ny);
    int size = nx * ny * sizeof(float);

    // printf("sumMatrixOnHost elapsed %fms\n", iElaps);

    float *hd_A, *hd_B, *hd_C, *h_D;
    cudaMallocManaged((void**)&hd_A, size);
    cudaMallocManaged((void**)&hd_B, size);
    cudaMallocManaged((void**)&hd_C, size);
    cudaMallocManaged((void**)&h_D, size);
    iStart = cpuSecond();
    initialData(hd_A, nx * ny);
    initialData(hd_B, nx * ny);
    iElaps = cpuSecond() - iStart;
    printf("initialization: \t %f sec\n", iElaps);

    iStart = cpuSecond();
    sumMatrixOnHost(hd_A, hd_B, h_D, nx, ny);
    iElaps = cpuSecond() - iStart;
    printf("sumMatrix on host:\t %f sec\n", iElaps);
    // cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    // cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    int dimx = 32;
    int dimy = 32;
    if (argc > 2) {
        dimx = atoi(argv[1]);
        dimy = atoi(argv[2]);
    }

    dim3 block(dimx, dimy);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
    CHECK(cudaDeviceSynchronize());
    iStart = cpuSecond();
    sumMatrix<<<grid, block>>>(hd_A, hd_B, hd_C, nx, ny);
    CHECK(cudaDeviceSynchronize());
    iElaps = cpuSecond() - iStart;
    printf("sumMatrix on GPU <<<(%d, %d), (%d, %d)>>> elapsed %fsec\n", grid.x, grid.y, block.x, block.y, iElaps);
    CHECK(cudaGetLastError());

    checkResult(h_D, hd_C, nx * ny);

    CHECK(cudaFree(hd_A));
    CHECK(cudaFree(hd_B));
    CHECK(cudaFree(hd_C));
    CHECK(cudaFree(h_D));

    CHECK(cudaDeviceReset());

    return 0;
}