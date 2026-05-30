
#include <cuda_runtime.h>
#include <stdio.h>


int main(){
    int numDevices = 0;
    // 获取GPU数量
    cudaGetDeviceCount(&numDevices);
    printf("num device:%d", numDevices);

    if(numDevices > 1){
        int maxMultiprocessors = 0;
        int maxDevice = 0;
        for (int device=0; device < numDevices; device ++){
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, device);
            // 流式多处理器（SM）数量
            if(maxMultiprocessors > prop.multiProcessorCount){
                maxMultiprocessors = prop.multiProcessorCount;
                maxDevice = device;
            }
        }
        cudaSetDevice(maxDevice);
    }

    return 0;
}