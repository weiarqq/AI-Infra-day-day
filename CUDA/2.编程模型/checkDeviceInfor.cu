#include <cuda_runtime.h>
#include <stdio.h>

int main(int argc, char **argv) {
    printf("%s Starting...\n", argv[0]);

    int deviceCount = 0;
    // 设备数量检测
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
    if (error_id != cudaSuccess) {
        printf("cudaGetDeviceCount returned %d\n-> %s\n",
            (int)error_id, cudaGetErrorString(error_id));
        printf("Result = FAIL\n");
        exit(EXIT_FAILURE);
    }

    if (deviceCount == 0) {
        printf("There are no available device(s) that support CUDA\n");
    } else {
        printf("Detected %d CUDA Capable device(s)\n", deviceCount);
    }

    int dev, driverVersion = 0, runtimeVersion = 0;

    dev = 0;
    cudaSetDevice(dev);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    // GPU 设备名称
    printf("Device %d: \"%s\"\n", dev, deviceProp.name);
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);
    // CUDA 版本
    printf("CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n",
        driverVersion/1000, (driverVersion%100)/10,
        runtimeVersion/1000, (runtimeVersion%100)/10);
    // CUDA 计算能力 数字越高，支持的并行计算功能越强（比如 8.6=RTX 30 系列，7.5=RTX 20 系列，9.0=RTX 40 系列
    printf("CUDA Capability Major/Minor version number:    %d.%d\n",
        deviceProp.major, deviceProp.minor);

    // 显存大小
    printf("Total amount of global memory:                 %.2f MBytes (%llu bytes)\n",
        (float)deviceProp.totalGlobalMem/pow(1024.0,3),
        (unsigned long long)deviceProp.totalGlobalMem);
    // GPU 核心频率
    printf("GPU Clock rate:                                %.0f MHz (%.2f GHz)\n",
        deviceProp.clockRate * 1e-3, deviceProp.clockRate * 1e-6);
    // 显存相关参数 显存时钟（MHz）越高越好
    printf("Memory Clock rate:                             %.0f Mhz\n",
        deviceProp.memoryClockRate * 1e-3);
    // 显存总线宽度（bit）越宽越好（总线越宽，显存带宽越高，读写数据越快）
    printf("Memory Bus Width:                              %d-bit\n",
        deviceProp.memoryBusWidth);
    if (deviceProp.l2CacheSize) {
        // L2 缓存大小
        printf("L2 Cache Size:                                 %d bytes\n",
            deviceProp.l2CacheSize);
    }
    // 纹理内存限制
    printf(" Max Texture Dimension Size (x,y,z)            1D=(%d), 2D=(%d,%d), 3D=(%d,%d,%d)\n",
        deviceProp.maxTexture1D, deviceProp.maxTexture2D[0],
        deviceProp.maxTexture2D[1], deviceProp.maxTexture3D[0], deviceProp.maxTexture3D[1],
        deviceProp.maxTexture3D[2]);
    // CUDA 中 “分层纹理” 的最大尺寸限制
    printf(" Max Layered Texture Size (dim) x layers      1D=(%d) x %d, 2D=(%d,%d) x %d\n",
        deviceProp.maxTexture1DLayered[0], deviceProp.maxTexture1DLayered[1],
        deviceProp.maxTexture2DLayered[0], deviceProp.maxTexture2DLayered[1],
        deviceProp.maxTexture2DLayered[2]);
    // 常量内存总量
    printf("Total amount of constant memory:               %lu bytes\n",
        deviceProp.totalConstMem);
    // 共享内存
    printf("Total amount of shared memory per block:       %lu bytes\n",
        deviceProp.sharedMemPerBlock);
    // 寄存器
    printf("Total registers available per block:           %d\n",
        deviceProp.regsPerBlock);
    // 线程束大小
    printf(" Warp size:                                    %d\n", deviceProp.warpSize);
    // 每个SM的最大线程束数量
    printf(" Maximum number of threads per multiprocessor: %d\n",
        deviceProp.maxThreadsPerMultiProcessor);
    // 每个块 线程数限制
    printf(" Maximum number of threads per block:          %d\n",
        deviceProp.maxThreadsPerBlock);
    // 每个块最大维度
    printf(" Max dimension size of a block:                %d x %d x %d\n",
        deviceProp.maxThreadsDim[0],
        deviceProp.maxThreadsDim[1],
        deviceProp.maxThreadsDim[2]);
    // 每个网格最大维度
    printf(" Max dimension size of a grid:                 %d x %d x %d\n",
        deviceProp.maxGridSize[0],
        deviceProp.maxGridSize[1],
        deviceProp.maxGridSize[2]);
    // 最大内存步长
    printf(" Maximum memory pitch:                         %lu bytes\n", deviceProp.memPitch);

    exit(EXIT_SUCCESS);
}