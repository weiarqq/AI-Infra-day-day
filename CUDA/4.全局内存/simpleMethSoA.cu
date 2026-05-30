#include "common.h"

#define LLEN 1 << 22

struct innerStruct {
    float x;
    float y;
};

struct innerArray {
    float x[LLEN];
    float y[LLEN];
};

void initialInnerStruct(innerArray* stru, int size)
{
    for (int i = 0; i < size; i++) {
        stru->x[i] = (float)(rand() & 0xFF) / 100.0f;
        stru->y[i] = (float)(rand() & 0xFF) / 100.0f;
    }
}

void testInnerStructOnHost(innerArray* stru, innerArray* res, int size)
{
    for (int i = 0; i < size; i++) {
        res->x[i] = stru->x[i] + 10.0f;
        res->y[i] = stru->y[i] + 10.0f;
    }
}

__global__ void testInnerStructOnDevice(innerArray* stru, innerArray* res, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        res->x[idx] = stru->x[idx] + 10.0f;
        res->y[idx] = stru->y[idx] + 10.0f;
    }
}
__global__ void warmup(innerArray* stru, innerArray* res, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        res->x[idx] = stru->x[idx] + 10.0f;
        res->y[idx] = stru->y[idx] + 10.0f;
    }
}

void checkInnerStruct(innerArray* hostRef, innerArray* gpuRef, const int N)
{
    double epsilon = 1.0E-8;
    bool match = 1;

    for (int i = 0; i < N; i++) {
        if (abs(hostRef->x[i] - gpuRef->x[i]) > epsilon) {
            match = 0;
            printf("different on %dth element: host %f gpu %f\n", i,
                hostRef->x[i], gpuRef->x[i]);
            break;
        }

        if (abs(hostRef->y[i] - gpuRef->y[i]) > epsilon) {
            match = 0;
            printf("different on %dth element: host %f gpu %f\n", i,
                hostRef->y[i], gpuRef->y[i]);
            break;
        }
    }

    if (!match)
        printf("Arrays do not match.\n\n");
}

int main(int argc, char** argv)
{
    // printf("%s Starting...\n", argv[0]);
    int dev = 0;
    double iStart, iElaps;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    // printf("Use Device: %d %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    int nElem = LLEN;
    int nBytes = sizeof(innerArray);
    printf("Vector size %d nbytes  %3.0f MB\n", nElem,
        (float)nBytes / (1024.0f * 1024.0f));

    innerArray *h_A, *h_S, *hd_S;
    h_A = new innerArray();
    h_S = new innerArray();
    hd_S = new innerArray();

    initialInnerStruct(h_A, nElem);

    testInnerStructOnHost(h_A, h_S, nElem);

    innerArray *d_A, *d_S;
    CHECK(cudaMalloc((innerArray**)&d_A, nBytes));
    CHECK(cudaMalloc((innerArray**)&d_S, nBytes));

    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));

    int blockSize = 128;
    if (argc > 1)
        blockSize = atoi(argv[1]);

    dim3 block(blockSize, 1);
    dim3 grid((nElem + block.x - 1) / block.x, 1);
    iStart = cpuSecond();
    warmup<<<grid, block>>>(d_A, d_S, nElem);
    CHECK(cudaDeviceSynchronize())
    iElaps = cpuSecond() - iStart;

    iStart = cpuSecond();
    testInnerStructOnDevice<<<grid, block>>>(d_A, d_S, nElem);
    CHECK(cudaDeviceSynchronize())
    printf("innerstruct <<< %3d, %3d >>> elapsed %f sec\n", grid.x, block.x,
        iElaps);
    iElaps = cpuSecond() - iStart;
    CHECK(cudaMemcpy(hd_S, d_S, nBytes, cudaMemcpyDeviceToHost));

    checkInnerStruct(h_S, hd_S, nElem);
    CHECK(cudaGetLastError());

    cudaFree(d_A);
    cudaFree(d_S);

    std::free(h_A);
    std::free(h_S);
    std::free(hd_S);

    cudaDeviceReset();
    return 0;
}