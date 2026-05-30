#include "common.h"
#include "omp.h"
#define N 300000
#define NSTREAM 4

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

    float elapsed_time = 0.0f;
    cudaEventRecord(start);
    omp_set_num_threads(n_streams);
#pragma omp parallel
    {
        int i = omp_get_thread_num();
        kernel_1<<<grid, block, 0, streams[i]>>>();
        kernel_2<<<grid, block, 0, streams[i]>>>();
        kernel_3<<<grid, block, 0, streams[i]>>>();
        kernel_4<<<grid, block, 0, streams[i]>>>();
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