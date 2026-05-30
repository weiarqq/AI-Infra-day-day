#include "common.h"

#define NSTREAM 4
#define BDIM 128

void sumArrayOnHost(float* A, float* B, float* C, const int N)
{
    for (int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }
}

__global__ void sumArrays(float* A, float* B, float* C, const int N)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        for (int i = 0; i < N; i++) {
            C[idx] = A[idx] + B[idx];
        }
    }
}

int main(int argc, char** argv)
{
    printf("> %s Starting...\n", argv[0]);

    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("> Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // check if device support hyper-q
    if (deviceProp.major < 3 || (deviceProp.major == 3 && deviceProp.minor < 5)) {
        if (deviceProp.concurrentKernels == 0) {
            printf("> GPU does not support concurrent kernel execution (SM 3.5 "
                   "or higher required)\n");
            printf("> CUDA kernel runs will be serialized\n");
        } else {
            printf("> GPU does not support HyperQ\n");
            printf("> CUDA kernel runs will have limited concurrency\n");
        }
    }

    printf("> Compute Capability %d.%d hardware with %d multi-processors\n",
        deviceProp.major, deviceProp.minor, deviceProp.multiProcessorCount);

    // set up max connectioin
    const char* iname = "CUDA_DEVICE_MAX_CONNECTIONS";
    setenv(iname, "1", 1);
    char* ivalue = getenv(iname);
    printf("> %s = %s\n", iname, ivalue);
    printf("> with streams = %d\n", NSTREAM);

    // set up data size of vectors
    int nElem = 1 << 18;
    printf("> vector size = %d\n", nElem);
    size_t nBytes = nElem * sizeof(float);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float itotal = 0.0f;

    float *h_A, *h_B, *hostRef, *gpuRef;
    CHECK(cudaHostAlloc((float**)&h_A, nBytes, cudaHostAllocDefault));
    CHECK(cudaHostAlloc((float**)&h_B, nBytes, cudaHostAllocDefault));
    CHECK(cudaHostAlloc((float**)&hostRef, nBytes, cudaHostAllocDefault));
    CHECK(cudaHostAlloc((float**)&gpuRef, nBytes, cudaHostAllocDefault));
    initialData(h_A, nElem);
    initialData(h_B, nElem);
    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    sumArrayOnHost(h_A, h_B, hostRef, nElem);

    float *d_A, *d_B, *d_C;
    CHECK(cudaMalloc((float**)&d_A, nBytes));
    CHECK(cudaMalloc((float**)&d_B, nBytes));
    CHECK(cudaMalloc((float**)&d_C, nBytes));

    cudaEventRecord(start, 0);
    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));
    cudaEventRecord(stop, 0);
    CHECK(cudaEventSynchronize(stop));
    float memcpy_h2d_time;
    cudaEventElapsedTime(&memcpy_h2d_time, start, stop);
    itotal += memcpy_h2d_time;

    dim3 block(BDIM);
    dim3 grid((nElem + block.x - 1) / block.x);

    cudaEventRecord(start, 0);
    sumArrays<<<grid, block>>>(d_A, d_B, d_C, nElem);
    cudaEventRecord(stop, 0);
    CHECK(cudaEventSynchronize(stop));
    float kernel_time;
    cudaEventElapsedTime(&kernel_time, start, stop);
    itotal += kernel_time;

    cudaEventRecord(start, 0);
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    cudaEventRecord(stop, 0);
    CHECK(cudaEventSynchronize(stop));
    float memcpy_d2h_time;
    cudaEventElapsedTime(&memcpy_d2h_time, start, stop);
    itotal += memcpy_d2h_time;
    printf("\n");
    printf("Measured timings (throughput):\n");
    printf(" Memcpy host to device\t: %f ms (%f GB/s)\n",
        memcpy_h2d_time, (nBytes * 1e-6) / memcpy_h2d_time);
    printf(" Memcpy device to host\t: %f ms (%f GB/s)\n",
        memcpy_d2h_time, (nBytes * 1e-6) / memcpy_d2h_time);
    printf(" Kernel\t\t\t: % f ms (%f GB/s)\n",
        kernel_time, (nBytes * 2e-6) / kernel_time);
    printf(" Total\t\t\t: %f ms (%f GB/s)\n",
        itotal, (nBytes * 2e-6) / itotal);

    int iElem = nElem / NSTREAM;
    int ibytes = iElem * sizeof(float);
    grid.x = (iElem + block.x - 1) / block.x;
    cudaStream_t streams[NSTREAM];
    for (int i = 0; i < NSTREAM; i++) {
        cudaStreamCreate(&streams[i]);
    }
    cudaEventRecord(start, 0);
    for (int i = 0; i < NSTREAM; i++) {
        int offset = i * iElem;
        cudaMemcpyAsync(&d_A[offset], &h_A[offset], ibytes, cudaMemcpyHostToDevice, streams[i]);
        cudaMemcpyAsync(&d_B[offset], &h_B[offset], ibytes, cudaMemcpyHostToDevice, streams[i]);
        sumArrays<<<grid, block, 0, streams[i]>>>(&d_A[offset], &d_B[offset], &d_C[offset], iElem);
        cudaMemcpyAsync(&gpuRef[offset], &d_C[offset], ibytes, cudaMemcpyDeviceToHost, streams[i]);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elaspsed_time;
    cudaEventElapsedTime(&elaspsed_time, start, stop);

    printf("\n");
    printf("Actual results from overlapped data transfers:\n");
    printf(" overlap with %d streams : %f ms (%f GB/s)\n", NSTREAM,
        elaspsed_time, (nBytes * 2e-6) / elaspsed_time);
    printf(" speedup                : %f \n",
        ((itotal - elaspsed_time) * 100.0f) / itotal);

    // check kernel error
    CHECK(cudaGetLastError());

    // check device results
    checkResult(hostRef, gpuRef, nElem);

    // free device global memory
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));

    // free host memory
    CHECK(cudaFreeHost(h_A));
    CHECK(cudaFreeHost(h_B));
    CHECK(cudaFreeHost(hostRef));
    CHECK(cudaFreeHost(gpuRef));

    // destroy events
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));

    // destroy streams
    for (int i = 0; i < NSTREAM; ++i) {
        CHECK(cudaStreamDestroy(streams[i]));
    }

    CHECK(cudaDeviceReset());
    return (0);
}