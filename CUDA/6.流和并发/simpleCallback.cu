#include "common.h"

#define N 300000
#define NSTREAM 4

void CUDART_CB my_callback(cudaStream_t stream, cudaError_t status, void* data)
{
    printf("callback from stream %d\n", *((int*)data));
}

__global__ void kernel_1()
{
    double sum = 0.0;

    for (int i = 0; i < N; i++) {
        sum = sum + tan(0.1) * tan(0.1);
    }
}
__global__ void kernel_2()
{
    double sum = 0.0;

    for (int i = 0; i < N; i++) {
        sum = sum + tan(0.1) * tan(0.1);
    }
}
__global__ void kernel_3()
{
    double sum = 0.0;

    for (int i = 0; i < N; i++) {
        sum = sum + tan(0.1) * tan(0.1);
    }
}
__global__ void kernel_4()
{
    double sum = 0.0;

    for (int i = 0; i < N; i++) {
        sum = sum + tan(0.1) * tan(0.1);
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
    const char* iname = "CUDA_DEVICE_MAX_CONNECTIONS";
    setenv(iname, "4", 1);

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

    int n_streams = NSTREAM;
    cudaStream_t* streams = (cudaStream_t*)malloc(n_streams * sizeof(cudaStream_t));
    for (int i = 0; i < n_streams; i++) {
        cudaStreamCreate(&streams[i]);
    }

    dim3 block(1);
    dim3 grid(1);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    int stream_ids[n_streams];
    float elapsed_time = 0.0f;
    cudaEventRecord(start);
    for (int i = 0; i < n_streams; i++) {
        stream_ids[i] = i;
        kernel_1<<<grid, block, 0, streams[i]>>>();
        kernel_2<<<grid, block, 0, streams[i]>>>();
        kernel_3<<<grid, block, 0, streams[i]>>>();
        kernel_4<<<grid, block, 0, streams[i]>>>();
        // cudaStreamAddCallback(streams[i], my_callback, &stream_ids[i], 0);
        // 每个kernel执行完时会执行callback，非同步
        cudaStreamAddCallback(streams[i], my_callback, (void*)(stream_ids + i), 0);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed_time, start, end);

    printf("Measured time for parallel execution = %.3fs\n",
        elapsed_time / 1000.0f);

    for (int i = 0; i < n_streams; i++) {
        cudaStreamDestroy(streams[i]);
    }
    free(streams);
    cudaEventDestroy(start);
    cudaEventDestroy(end);

    CHECK(cudaDeviceReset());
    return 0;
}