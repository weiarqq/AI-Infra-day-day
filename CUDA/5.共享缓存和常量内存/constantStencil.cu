#include "common.h"

#define RADIUS 4
#define BDIM 32

__constant__ float coef[RADIUS + 1];

#define a0 0.00000f
#define a1 0.80000f
#define a2 -0.20000f
#define a3 0.03809f
#define a4 -0.00357f

const float h_coef[] = { a0, a1, a2, a3, a4 };

void setup_coef_constant(void)
{
    CHECK(cudaMemcpyToSymbol(coef, h_coef, (RADIUS + 1) * sizeof(float)));
}

void cpu_stencil_1d(float* in, float* out, int isize)
{
    for (int i = RADIUS; i <= isize; i++) {
        float tmp = 0.0f;
        for (int k = 1; k <= RADIUS; k++) {
            tmp += h_coef[k] * (in[i + k] - in[i - k]);
        }
        out[i] = tmp;
    }
}

void printData(float* in, const int size)
{
    for (int i = RADIUS; i < size; i++) {
        printf("%f ", in[i]);
    }

    printf("\n");
}

__global__ void stencil_1d(float* in, float* out)
{
    __shared__ float smem[BDIM + 2 * RADIUS];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int sidx = threadIdx.x + RADIUS;

    smem[sidx] = in[idx];

    if (threadIdx.x < RADIUS) {
        smem[sidx - RADIUS] = in[idx - RADIUS];
        smem[sidx + BDIM] = in[idx + BDIM];
    }

    __syncthreads();

    float tmp = 0.0f;
#pragma unroll
    for (int i = 1; i <= RADIUS; i++) {
        tmp += coef[i] * (smem[sidx + i] - smem[sidx - i]);
    }

    out[idx] = tmp;
}
void checkResultlocal(float* hostRef, float* gpuRef, const int size)
{
    double epsilon = 1.0E-5;
    bool match = 1;

    for (int i = RADIUS; i < size; i++) {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon) {
            match = 0;
            printf("different on %dth element: host %f gpu %f\n", i, hostRef[i],
                gpuRef[i]);
            break;
        }
    }

    if (!match)
        printf("Arrays do not match.\n\n");
}
int main(int argc, char** argv)
{
    printf("%s Starting...\n", argv[0]);
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Use Device: %d %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    int isize = 1 << 24;

    size_t nBytes = (isize + 2 * RADIUS) * sizeof(float);
    printf("array size: %d ", isize);

    bool iprint = 0;

    float* h_in = (float*)malloc(nBytes);
    float* hostRef = (float*)malloc(nBytes);
    float* gpuRef = (float*)malloc(nBytes);

    float *d_in, *d_out;
    CHECK(cudaMalloc((float**)&d_in, nBytes));
    CHECK(cudaMalloc((float**)&d_out, nBytes));

    initialData(h_in, isize + 2 * RADIUS);

    CHECK(cudaMemcpy(d_in, h_in, nBytes, cudaMemcpyHostToDevice));

    setup_coef_constant();

    cudaDeviceProp info;
    CHECK(cudaGetDeviceProperties(&info, 0));
    dim3 block(BDIM, 1);
    dim3 grid(isize / block.x, 1);
    printf("(grid, block) %d,%d \n ", grid.x, block.x);

    // Launch stencil_1d() kernel on GPU
    stencil_1d<<<grid, block>>>(d_in + RADIUS, d_out + RADIUS);

    // Copy result back to host
    CHECK(cudaMemcpy(gpuRef, d_out, nBytes, cudaMemcpyDeviceToHost));

    // apply cpu stencil
    cpu_stencil_1d(h_in, hostRef, isize);

    // check results
    checkResultlocal(hostRef, gpuRef, isize);

    // print out results
    if (iprint) {
        printData(gpuRef, isize);
        printData(hostRef, isize);
    }

    // Cleanup
    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_out));
    free(h_in);
    free(hostRef);
    free(gpuRef);

    // reset device
    CHECK(cudaDeviceReset());
    return EXIT_SUCCESS;
}