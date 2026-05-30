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

    float *h_A, *h_B, *h_Hs, *h_Ds;
    int size = nx * ny * sizeof(float);

    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_Hs = (float*)malloc(size);
    h_Ds = (float*)malloc(size);

    iStart = cpuSecond();
    initialData(h_A, nx * ny);
    initialData(h_B, nx * ny);
    iElaps = cpuSecond() - iStart;
    printf("initialization: \t %f sec\n", iElaps);

    memset(h_Hs, 0, size);
    memset(h_Ds, 0, size);

    iStart = cpuSecond();
    sumMatrixOnHost(h_A, h_B, h_Hs, nx, ny);
    iElaps = cpuSecond() - iStart;
    printf("sumMatrix On host elapsed %fms\n", iElaps);

    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
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
    sumMatrix<<<grid, block>>>(d_A, d_B, d_C, nx, ny);
    CHECK(cudaDeviceSynchronize());
    iElaps = cpuSecond() - iStart;
    printf("sumMatrix on GPU <<<(%d, %d), (%d, %d)>>> elapsed %fsec\n", grid.x, grid.y, block.x, block.y, iElaps);
    CHECK(cudaGetLastError());

    CHECK(cudaMemcpy(h_Ds, d_C, size, cudaMemcpyDeviceToHost));
    checkResult(h_Hs, h_Ds, nx * ny);

    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));

    free(h_A);
    free(h_B);
    free(h_Hs);
    free(h_Ds);

    CHECK(cudaDeviceReset());

    return 0;
}