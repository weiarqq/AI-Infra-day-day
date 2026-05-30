#include "common.h"
#include <cstdlib>
#include <cstring>

#define BDIMX 32
#define BDIMY 16

#define IPAD 1

void transposeHost(float* out, float* in, const int nx, const int ny)
{
    for (int iy = 0; iy < ny; iy++) {
        for (int ix = 0; ix < nx; ix++) {
            out[ix * ny + iy] = in[iy * nx + ix];
        }
    }
}

__global__ void naiveGmem(float* out, float* in, const int nx, const int ny)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < nx && idy < ny) {
        out[idx * ny + idy] = in[idy * nx + idx];
    }
}

__global__ void copyGmem(float* out, float* in, const int nx, const int ny)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < nx && idy < ny) {
        out[idy * nx + idx] = in[idy * nx + idx];
    }
}

__global__ void transposeSmem(float* out, float* in, const int nx, const int ny)
{
    // 1. copy in矩阵到共享内存title 按行读 按行写
    // 2. 找到当前线程处理的是第几个元素  bidx
    // 3. 找到该元素转置后在 当前线程块对应矩阵 的坐标 icol irow
    // 4. 找到应该填入out的位置, out的位置对应的是全局  to
    __shared__ float tile[BDIMY][BDIMX];
    unsigned int ix, iy, ti, to;

    // in[iy][ix]来取数的 行优先
    ix = blockIdx.x * blockDim.x + threadIdx.x;
    iy = blockIdx.y * blockDim.y + threadIdx.y;
    ti = iy * nx + ix;

    unsigned int bidx, irow, icol;
    bidx = threadIdx.y * blockDim.x + threadIdx.x; // 在当前线程块的位置
    irow = bidx / blockDim.y; // 在当前线程块转置后的位置 ix->icol iy->irow
    icol = bidx % blockDim.y;
    ix = blockIdx.y * blockDim.y + icol; // 映射到全局的位置，而不是当前线程块，将in转置后的矩阵 找到对应坐标
    iy = blockIdx.x * blockDim.x + irow;
    to = iy * ny + ix;

    if (ix < nx && iy < ny) {

        tile[threadIdx.y][threadIdx.x] = in[ti]; // tile 是按照 y,x来写入的 in也是按照in[threadIdx.y][threadIdx.x]来取数的 行优先

        __syncthreads();

        out[to] = tile[icol][irow]; // 所以需要按照 y,x来读取
    }
}

__global__ void transposeSmemUnroll(float* out, float* in, const int nx, const int ny)
{
    __shared__ float tile[BDIMY * (BDIMX * 2 + IPAD)];
    unsigned int ix, iy, ti, to;

    ix = 2 * blockIdx.x * blockDim.x + threadIdx.x;
    iy = blockIdx.y * blockDim.y + threadIdx.y;
    ti = iy * nx + ix;

    unsigned int bidx, irow, icol;
    bidx = threadIdx.y * blockDim.x + threadIdx.x;
    irow = bidx / blockDim.y;
    icol = bidx % blockDim.y;

    unsigned int ix2 = blockIdx.y * blockDim.y + icol;
    unsigned int iy2 = 2 * blockIdx.x * blockDim.x + irow;
    to = iy2 * ny + ix2;

    if (ix + blockDim.x < nx && iy < ny) {
        unsigned int row_idx = threadIdx.y * (2 * BDIMX + IPAD) + threadIdx.x;
        tile[row_idx] = in[ti];
        tile[row_idx + BDIMX] = in[ti + BDIMX];
        __syncthreads();
        unsigned int col_idx = icol * (blockDim.x * 2 + IPAD) + irow;
        out[to] = tile[col_idx]; // 所以需要按照 y,x来读取
        out[to + ny * BDIMX] = tile[col_idx + BDIMX];
    }
}

int main(int argc, char** argv)
{
    printf("%s Starting...\n", argv[0]);
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Use Device: %d %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    int nx = 1 << 12;
    int ny = 1 << 12;
    size_t nbytes = nx * ny * sizeof(float);
    float* h_idata = (float*)malloc(nbytes);
    float* h_odata = (float*)malloc(nbytes);
    float* hd_odata = (float*)malloc(nbytes);

    initialData(h_idata, nx * ny);
    memset(h_odata, 0, nbytes);

    float *d_idata, *d_odata;
    CHECK(cudaMalloc((float**)&d_idata, nbytes));
    CHECK(cudaMalloc((float**)&d_odata, nbytes));

    CHECK(cudaMemcpy(d_idata, h_idata, nbytes, cudaMemcpyHostToDevice));
    int bx = 32;
    int by = 16;
    int iprint = 0;
    if (argc > 1)
        iprint = atoi(argv[1]);

    if (argc > 2)
        bx = atoi(argv[2]);

    if (argc > 3)
        by = atoi(argv[3]);
    dim3 block(bx, by);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    CHECK(cudaMemset(d_odata, 0, nbytes));
    naiveGmem<<<grid, block>>>(d_odata, d_idata, nx, ny);
    cudaDeviceSynchronize();
    cudaMemcpy(hd_odata, d_odata, nbytes, cudaMemcpyDeviceToHost);
    if (iprint)
        printData("naiveGmem:", hd_odata, nx * ny);

    CHECK(cudaMemset(d_odata, 0, nbytes));
    transposeSmem<<<grid, block>>>(d_odata, d_idata, nx, ny);
    cudaDeviceSynchronize();
    cudaMemcpy(hd_odata, d_odata, nbytes, cudaMemcpyDeviceToHost);
    if (iprint)
        printData("transposeSmem:", hd_odata, nx * ny);

    CHECK(cudaMemset(d_odata, 0, nbytes));
    transposeSmemUnroll<<<grid, block>>>(d_odata, d_idata, nx, ny);
    cudaDeviceSynchronize();
    cudaMemcpy(hd_odata, d_odata, nbytes, cudaMemcpyDeviceToHost);
    if (iprint)
        printData("transposeSmemUnroll:", hd_odata, nx * ny);

    CHECK(cudaFree(d_idata));
    CHECK(cudaFree(d_odata));

    free(h_idata);
    free(h_odata);

    cudaDeviceReset();

    return 0;
}