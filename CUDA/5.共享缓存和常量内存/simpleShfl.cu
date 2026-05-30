#include "common.h"
#define BDIMX 16
#define SEGM 4
void printData(int* in, const int size)
{
    for (int i = 0; i < size; i++) {
        printf("%2d ", in[i]);
    }

    printf("\n");
}

__global__ void test_shfl_broadcast(int* d_out, int* d_in, int const srcLane)
{
    int value = d_in[threadIdx.x];
    value = __shfl_sync(0xffffffff, value, srcLane, BDIMX);
    d_out[threadIdx.x] = value;
}

__global__ void test_shfl_up(int* d_out, int* d_in, unsigned int const delta)
{
    int value = d_in[threadIdx.x];
    value = __shfl_up_sync(0xffffffff, value, delta, BDIMX);
    d_out[threadIdx.x] = value;
}

__global__ void test_shfl_down(int* d_out, int* d_in, unsigned int const delta)
{
    int value = d_in[threadIdx.x];
    value = __shfl_down_sync(0xffffffff, value, delta, BDIMX);
    d_out[threadIdx.x] = value;
}
__global__ void test_shfl_wrap(int* d_out, int* d_in, unsigned int const offset)
{
    int value = d_in[threadIdx.x];
    // threadIdx.x+offset 超出BDIMX后 shuffleID = (threadIdx.x+offset)%width
    value = __shfl_sync(0xffffffff, value, threadIdx.x + offset, BDIMX);
    d_out[threadIdx.x] = value;
}
// 实现了两个线程之间的蝴蝶寻址模式
__global__ void test_shfl_xor(int* d_out, int* d_in, unsigned int const mask)
{
    int value = d_in[threadIdx.x];
    value = __shfl_xor_sync(0xffffffff, value, mask, BDIMX);
    d_out[threadIdx.x] = value;
}

// 跨 SEGM 交换数据
__global__ void test_shfl_xor_array(int* d_out, int* d_in, int const mask)
{
    int idx = threadIdx.x * SEGM;
    int value[SEGM];

    for (int i = 0; i < SEGM; i++)
        value[i] = d_in[idx + i];

    value[0] = __shfl_xor_sync(0xffffffff, value[0], mask, BDIMX); // 0^1= 线程1 线程1的 value[0] = d_in[4]
    // printf("%d=0=>%d\n", threadIdx.x, value[0]);
    value[1] = __shfl_xor_sync(0xffffffff, value[1], mask, BDIMX); // delta
    // printf("%d=1=>%d\n", threadIdx.x, value[1]);

    value[2] = __shfl_xor_sync(0xffffffff, value[2], mask, BDIMX);
    // printf("%d=2=>%d\n", threadIdx.x, value[2]);

    value[3] = __shfl_xor_sync(0xffffffff, value[3], mask, BDIMX);
    // printf("%d=3=>%d\n", threadIdx.x, value[3]);

    for (int i = 0; i < SEGM; i++)
        d_out[idx + i] = value[i];
}

__inline__ __device__ void swap(int* value, int laneIdx, int mask, int firstIdx, int secondIdx)
{
    bool pred = ((laneIdx / mask + 1) == 1);

    if (pred) {
        int tmp = value[firstIdx];
        value[firstIdx] = value[secondIdx];
        value[secondIdx] = tmp;
    }

    value[secondIdx] = __shfl_xor_sync(0xffffffff, value[secondIdx], mask, BDIMX);

    if (pred) {
        int tmp = value[firstIdx];
        value[firstIdx] = value[secondIdx];
        value[secondIdx] = tmp;
    }
}

__global__ void test_shfl_swap(int* d_out, int* d_in, int const mask, int firstIdx, int secondIdx)
{
    int idx = threadIdx.x * SEGM;
    int value[SEGM];

    for (int i = 0; i < SEGM; i++)
        value[i] = d_in[idx + i];

    swap(value, threadIdx.x, mask, firstIdx, secondIdx);

    for (int i = 0; i < SEGM; i++)
        d_out[idx + i] = value[i];
}

int main(int argc, char** argv)
{
    printf("%s Starting...\n", argv[0]);
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Use Device: %d %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));
    bool iPrintout = 1;

    int nElem = BDIMX;
    int* h_inData = new int[BDIMX];
    int* h_outData = new int[BDIMX];

    for (int i = 0; i < nElem; i++)
        h_inData[i] = i;

    if (iPrintout) {
        printf("initialData\t\t: ");
        printData(h_inData, nElem);
    }

    size_t nBytes = nElem * sizeof(int);
    int *d_inData, *d_outData;
    CHECK(cudaMalloc((int**)&d_inData, nBytes));
    CHECK(cudaMalloc((int**)&d_outData, nBytes));

    CHECK(cudaMemcpy(d_inData, h_inData, nBytes, cudaMemcpyHostToDevice));

    int block = BDIMX;

    test_shfl_broadcast<<<1, block>>>(d_outData, d_inData, 2);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(h_outData, d_outData, nBytes, cudaMemcpyDeviceToHost));

    if (iPrintout) {
        printf("shfl bcast\t\t\t\t: ");
        printData(h_outData, nElem);
    }

    test_shfl_up<<<1, block>>>(d_outData, d_inData, 2);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(h_outData, d_outData, nBytes, cudaMemcpyDeviceToHost));

    if (iPrintout) {
        printf("shfl up\t\t: ");
        printData(h_outData, nElem);
    }
    test_shfl_down<<<1, block>>>(d_outData, d_inData, 2);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(h_outData, d_outData, nBytes, cudaMemcpyDeviceToHost));

    if (iPrintout) {
        printf("shfl down\t\t: ");
        printData(h_outData, nElem);
    }

    test_shfl_wrap<<<1, block>>>(d_outData, d_inData, 2);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(h_outData, d_outData, nBytes, cudaMemcpyDeviceToHost));

    if (iPrintout) {
        printf("shfl wrap\t\t: ");
        printData(h_outData, nElem);
    }

    test_shfl_xor<<<1, block>>>(d_outData, d_inData, 1);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(h_outData, d_outData, nBytes, cudaMemcpyDeviceToHost));

    if (iPrintout) {
        printf("shfl xor\t\t: ");
        printData(h_outData, nElem);
    }
    test_shfl_xor_array<<<1, block / SEGM>>>(d_outData, d_inData, 1);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(h_outData, d_outData, nBytes, cudaMemcpyDeviceToHost));

    if (iPrintout) {
        printf("shfl xor array\t\t: ");
        printData(h_outData, nElem);
    }

    test_shfl_swap<<<1, block / SEGM>>>(d_outData, d_inData, 1, 0, 3);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(h_outData, d_outData, nBytes, cudaMemcpyDeviceToHost));

    if (iPrintout) {
        printf("shfl swap\t\t: ");
        printData(h_outData, nElem);
    }

    CHECK(cudaFree(d_inData));
    CHECK(cudaFree(d_outData));
    CHECK(cudaDeviceReset(););

    return EXIT_SUCCESS;
}