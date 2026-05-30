#include "common.h"
#include <cstdlib>

void transposeHost(float* out, float* in, const int nx, const int ny)
{
    for (int iy = 0; iy < ny; iy++) {
        for (int ix = 0; ix < nx; ix++) {
            out[ix * ny + iy] = in[iy * nx + ix];
        }
    }
}

__global__ void warmup(float* out, float* in, const int nx, const int ny)
{
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    if (ix < nx && iy < ny) {
        out[iy * nx + ix] = in[iy * nx + ix];
    }
}

__global__ void copyRow(float* out, float* in, const int nx, const int ny)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx < nx && idy < ny) {
        out[idy * nx + idx] = in[idy * nx + idx];
    }
}
__global__ void copyCol(float* out, float* in, const int nx, const int ny)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx < nx && idy < ny) {
        out[idx * ny + idy] = in[idx * ny + idy];
    }
}

__global__ void transposeNaiveRow(float* out, float* in, const int nx, const int ny)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx < nx && idy < ny) {
        out[idx * ny + idy] = in[idy * nx + idx];
    }
}

__global__ void transposeUroll4Row(float* out, float* in, const int nx, const int ny)
{
    unsigned int idx = blockIdx.x * blockDim.x * 4 + threadIdx.x;
    unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int iidx = idy * nx + idx;
    unsigned int oidx = idx * ny + idy;

    if (idx + 3 * blockDim.x < nx && idy < ny) {
        out[oidx] = in[iidx];
        out[oidx + ny * blockDim.x] = in[iidx + blockDim.x]; // iidx 跳转 blockDim.x步 转置后跳转 ny*blockDim.x步，iidx走一步，oidx走ny步
        out[oidx + 2 * ny * blockDim.x] = in[iidx + 2 * blockDim.x];
        out[oidx + 3 * ny * blockDim.x] = in[iidx + 3 * blockDim.x];
    }
}

__global__ void transposeDiagonalRow(float* out, float* in, const int nx, const int ny)
{
    unsigned int block_y = blockIdx.x;
    unsigned int block_x = (blockIdx.x + blockIdx.y) % gridDim.x;

    unsigned int idx = block_x * blockDim.x + threadIdx.x;
    unsigned int idy = block_y * blockDim.y + threadIdx.y;
    if (idx < nx && idy < ny) {
        out[idx * ny + idy] = in[idy * nx + idx];
    }
}

__global__ void transposeNaiveCol(float* out, float* in, const int nx, const int ny)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx < nx && idy < ny) {
        out[idy * nx + idx] = in[idx * ny + idy];
    }
}

__global__ void transposeUroll4Col(float* out, float* in, const int nx, const int ny)
{
    unsigned int idx = blockIdx.x * blockDim.x * 4 + threadIdx.x;
    unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int iidx = idx * ny + idy;
    unsigned int oidx = idy * nx + idx;
    if (idx < nx && idy < ny) {
        out[oidx] = in[iidx];
        out[oidx + blockDim.x] = in[iidx + ny * blockDim.x];
        out[oidx + 2 * blockDim.x] = in[iidx + ny * 2 * blockDim.x];
        out[oidx + 3 * blockDim.x] = in[iidx + ny * 3 * blockDim.x];
    }
}

__global__ void transposeDiagonalCol(float* out, float* in, const int nx, const int ny)
{
    unsigned int block_y = blockIdx.x;
    unsigned int block_x = (blockIdx.x + blockIdx.y) % gridDim.x;

    unsigned int idx = block_x * blockDim.x + threadIdx.x;
    unsigned int idy = block_y * blockDim.y + threadIdx.y;
    if (idx < nx && idy < ny) {
        out[idy * nx + idx] = in[idx * ny + idy];
    }
}

// main functions
int main(int argc, char** argv)
{
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("%s starting transpose at ", argv[0]);
    printf("device %d: %s ", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));
    double iStart, iElaps;

    // set up array size 2048
    int nx = 1 << 11;
    int ny = 1 << 11;

    // select a kernel and block size
    int iKernel = 0;
    int blockx = 16;
    int blocky = 16;

    if (argc > 1)
        iKernel = atoi(argv[1]);

    if (argc > 2)
        blockx = atoi(argv[2]);

    if (argc > 3)
        blocky = atoi(argv[3]);

    if (argc > 4)
        nx = atoi(argv[4]);

    if (argc > 5)
        ny = atoi(argv[5]);

    printf(" with matrix nx %d ny %d with kernel %d\n", nx, ny, iKernel);
    size_t nBytes = nx * ny * sizeof(float);

    dim3 block(blockx, blocky);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
    float* h_A = (float*)std::malloc(nBytes);
    float* h_S = (float*)std::malloc(nBytes);
    float* hd_S = (float*)std::malloc(nBytes);

    initialData(h_A, nx * ny);

    transposeHost(h_S, h_A, nx, ny);

    float *d_A, *d_S;
    CHECK(cudaMalloc((float**)&d_A, nBytes));
    CHECK(cudaMalloc((float**)&d_S, nBytes));

    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
    iStart = cpuSecond();
    warmup<<<grid, block>>>(d_S, d_A, nx, ny);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    printf("warmup         elapsed %f sec\n", iElaps);
    CHECK(cudaGetLastError());
    void (*kernel)(float*, float*, int, int);
    const char* kernelName;
    switch (iKernel) {
    case 0:
        kernel = &copyRow;
        kernelName = "CopyRow    ";
        break;
    case 1:
        kernel = &copyCol;
        kernelName = "CopyCol    ";
        break;
    case 2:
        kernel = &transposeNaiveRow;
        kernelName = "transposeNaiveRow    ";
        break;
    case 3:
        kernel = &transposeNaiveCol;
        kernelName = "transposeNaiveCol    ";
        break;
    case 4:
        kernel = &transposeUroll4Row;
        kernelName = "transposeUroll4Row    ";
        grid.x = (nx + 4 * block.x - 1) / 4 * block.x;
        break;
    case 5:
        kernel = &transposeUroll4Col;
        kernelName = "transposeUroll4Col    ";
        grid.x = (nx + 4 * block.x - 1) / 4 * block.x;
        break;
    case 6:
        kernel = &transposeDiagonalRow;
        kernelName = "transposeDiagonalRow    ";
        break;
    case 7:
        kernel = &transposeDiagonalCol;
        kernelName = "transposeDiagonalCol    ";
        break;
    }
    iStart = cpuSecond();
    kernel<<<grid, block>>>(d_S, d_A, nx, ny);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;

    float ibnd = 2 * nx * ny * sizeof(float) / 1e9 / iElaps; // 每秒处理的数据量(读和写) GB/s
    printf("%s elapsed %f sec <<< grid (%d,%d) block (%d,%d)>>> effective "
           "bandwidth %f GB\n",
        kernelName, iElaps, grid.x, grid.y, block.x,
        block.y, ibnd);
    CHECK(cudaGetLastError());

    if (iKernel > 1) {
        cudaMemcpy(hd_S, d_S, nBytes, cudaMemcpyDeviceToHost);
        checkResult(h_S, hd_S, nx * ny);
    }

    cudaFree(d_A);
    cudaFree(d_S);
    free(h_A);
    free(h_S);
    free(hd_S);

    cudaDeviceReset();
    return 0;
}