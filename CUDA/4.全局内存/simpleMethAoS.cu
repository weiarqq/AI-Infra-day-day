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

void initialInnerStruct(innerStruct* stru, int size)
{
    for (int i = 0; i < size; i++) {
        stru[i].x = (float)(rand() & 0xFF) / 100.0f;
        stru[i].y = (float)(rand() & 0xFF) / 100.0f;
    }
}

void testInnerStructOnHost(innerStruct* stru, innerStruct* res, int size)
{
    for (int i = 0; i < size; i++) {
        res[i].x = stru[i].x + 10.0f;
        res[i].y = stru[i].y + 10.0f;
    }
}

__global__ void testInnerStructOnDevice(innerStruct* stru, innerStruct* res, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        res[idx].x = stru[idx].x + 10.0f;
        res[idx].y = stru[idx].y + 10.0f;
    }
}
__global__ void warmup(innerStruct* stru, innerStruct* res, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        res[idx].x = stru[idx].x + 10.0f;
        res[idx].y = stru[idx].y + 10.0f;
    }
}

void checkInnerStruct(innerStruct* hostRef, innerStruct* gpuRef, const int N)
{
    double epsilon = 1.0E-8;
    bool match = 1;

    for (int i = 0; i < N; i++) {
        if (abs(hostRef[i].x - gpuRef[i].x) > epsilon) {
            match = 0;
            printf("different on %dth element: host %f gpu %f\n", i,
                hostRef[i].x, gpuRef[i].x);
            break;
        }

        if (abs(hostRef[i].y - gpuRef[i].y) > epsilon) {
            match = 0;
            printf("different on %dth element: host %f gpu %f\n", i,
                hostRef[i].y, gpuRef[i].y);
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
    int nBytes = nElem * sizeof(innerStruct);
    printf("Vector size %d nbytes  %3.0f MB\n", nElem,
        (float)nBytes / (1024.0f * 1024.0f));

    innerStruct *h_A, *h_S, *hd_S;
    h_A = new innerStruct[nElem];
    h_S = new innerStruct[nElem];
    hd_S = new innerStruct[nElem];

    initialInnerStruct(h_A, nElem);

    testInnerStructOnHost(h_A, h_S, nElem);

    innerStruct *d_A, *d_S;
    CHECK(cudaMalloc((innerStruct**)&d_A, nBytes));
    CHECK(cudaMalloc((innerStruct**)&d_S, nBytes));

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