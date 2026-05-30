#include "common.h"
#include <cuda_runtime.h>
#include <stdio.h>



int recursiveReduce(int* data, int size){

    if(size == 1) return data[0];

    const int stride = size / 2;

    for(int i = 0; i< stride; i++){
        data[i] += data[i+stride];
    }

    return recursiveReduce(data, stride);

}
// 并行归约中的分化 每0 2 4 6 8 ...线程起作用
__global__ void reduceNeighbored(int *g_idata, int *g_odata, unsigned int n){

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx > n) return;

    // 找到当前block处理数组段中的位置
    int *idata = g_idata + blockIdx.x * blockDim.x;


    // 相邻元素相加，stride翻倍 第一轮1 第二轮 2 第三轮 4
    for(int stride = 1; stride < blockDim.x; stride*=2){
        if(tid%(2*stride) == 0){
            idata[tid] += idata[tid+stride]; // 2*stride 实际包括的范围是 2*stride
        }
        __syncthreads();
    }
    // 该数组段，即该block需要处理的数组段 相加的结果 放在idata[0], 然后将结果放到g_odata对应的block.x的位置
    if(tid == 0) g_odata[blockIdx.x] = idata[0];
}

// 并行归约中的分化 线程按序计算0 1 2 3 4
__global__ void reduceNeighboredLess(int *g_idata, int *g_odata, unsigned int n){

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx > n) return;

    // 找到当前block处理数组段中的位置
    int *idata = g_idata + blockIdx.x * blockDim.x;


    // 相邻元素相加，stride翻倍 第一轮1 第二轮 2 第三轮 4
    for(int stride = 1; stride < blockDim.x; stride*=2){
        int index = 2*stride*tid; // 0 1 2 3
        if(index < blockDim.x){
            idata[index] += idata[index+stride]; // 实际包括的范围是 2*stride,相邻元素，一次处理两个元素
        }
        __syncthreads();
    }

    // 该数组段，即该block需要处理的数组段 相加的结果 放在idata[0], 然后将结果放到g_odata对应的block.x的位置
    if(tid == 0) g_odata[blockIdx.x] = idata[0];
}


__global__ void reduceNeighboredLeaved(int *g_idata, int *g_odata, unsigned int n){

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx > n) return;

    // 找到当前block处理数组段中的位置
    int *idata = g_idata + blockIdx.x * blockDim.x;

    for(int stride = blockDim.x/2;stride >0; stride >>= 1){
        if(tid < stride){
            idata[tid] += idata[tid+stride];
        }
        __syncthreads();
    }
    // 该数组段，即该block需要处理的数组段 相加的结果 放在idata[0], 然后将结果放到g_odata对应的block.x的位置
    if(tid == 0) g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceUnrolling2(int *g_idata, int *g_odata, unsigned int n){

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // 找到当前block处理数组段中的位置
    int *idata = g_idata + blockIdx.x * blockDim.x * 2;
    
    if(idx+blockDim.x < n) g_idata[idx] += g_idata[idx+blockDim.x];

    __syncthreads();
    for(int stride = blockDim.x/2;stride >0; stride >>= 1){
        if(tid < stride){
            idata[tid] += idata[tid+stride];
        }
        __syncthreads();
    }
    // 该数组段，即该block需要处理的数组段 相加的结果 放在idata[0], 然后将结果放到g_odata对应的block.x的位置
    if(tid == 0) g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceUnrolling4(int *g_idata, int *g_odata, unsigned int n){

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x * 4 + threadIdx.x;

    // 找到当前block处理数组段中的位置
    int *idata = g_idata + blockIdx.x * blockDim.x * 4;
    
    if(idx+3*blockDim.x < n){
        int a = g_idata[idx];
        int b = g_idata[idx+blockDim.x];
        int c = g_idata[idx+2*blockDim.x];
        int d = g_idata[idx+3*blockDim.x];
        g_idata[idx] = a + b + c + d;
    } 
    __syncthreads();
    for(int stride = blockDim.x/2;stride >0; stride >>= 1){
        if(tid < stride){
            idata[tid] += idata[tid+stride];
        }
        __syncthreads();
    }
    // 该数组段，即该block需要处理的数组段 相加的结果 放在idata[0], 然后将结果放到g_odata对应的block.x的位置
    if(tid == 0) g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceUnrolling8(int *g_idata, int *g_odata, unsigned int n){

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

    // 找到当前block处理数组段中的位置
    int *idata = g_idata + blockIdx.x * blockDim.x * 8;
    
    if(idx+7*blockDim.x < n){
        int a = g_idata[idx];
        int b = g_idata[idx+blockDim.x];
        int c = g_idata[idx+2*blockDim.x];
        int d = g_idata[idx+3*blockDim.x];
        int e = g_idata[idx+4*blockDim.x];
        int f = g_idata[idx+5*blockDim.x];
        int g = g_idata[idx+6*blockDim.x];
        int h = g_idata[idx+7*blockDim.x];
        g_idata[idx] = a + b + c + d + e + f + g + h;
    } 
    __syncthreads();
    for(int stride = blockDim.x/2;stride >0; stride >>= 1){
        if(tid < stride){
            idata[tid] += idata[tid+stride];
        }
        __syncthreads();
    }
    // 该数组段，即该block需要处理的数组段 相加的结果 放在idata[0], 然后将结果放到g_odata对应的block.x的位置
    if(tid == 0) g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceCompleteUnrollWarps8(int *g_idata, int *g_odata, unsigned int n){

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

    // 找到当前block处理数组段中的位置
    int *idata = g_idata + blockIdx.x * blockDim.x * 8;
    
    if(idx+7*blockDim.x < n){
        int a = g_idata[idx];
        int b = g_idata[idx+blockDim.x];
        int c = g_idata[idx+2*blockDim.x];
        int d = g_idata[idx+3*blockDim.x];
        int e = g_idata[idx+4*blockDim.x];
        int f = g_idata[idx+5*blockDim.x];
        int g = g_idata[idx+6*blockDim.x];
        int h = g_idata[idx+7*blockDim.x];
        g_idata[idx] = a + b + c + d + e + f + g + h;
    } 
    __syncthreads();
    if(blockDim.x >=1024 && tid <512) idata[tid] += idata[tid + 512];
    __syncthreads();
    if(blockDim.x >=512 && tid <256) idata[tid] += idata[tid + 256];
    __syncthreads();
    if(blockDim.x >=256 && tid <128) idata[tid] += idata[tid + 128];
    __syncthreads();
    if(blockDim.x >=128 && tid <64) idata[tid] += idata[tid + 64];
    __syncthreads();
    if(tid < 32){
        volatile int *vmem = idata;
        vmem[tid] += vmem[tid+32];
        vmem[tid] += vmem[tid+16];
        vmem[tid] += vmem[tid+8];
        vmem[tid] += vmem[tid+4];
        vmem[tid] += vmem[tid+2];
        vmem[tid] += vmem[tid+1];
    }
    // 该数组段，即该block需要处理的数组段 相加的结果 放在idata[0], 然后将结果放到g_odata对应的block.x的位置
    if(tid == 0) g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceUnrollWarps8(int *g_idata, int *g_odata, unsigned int n){

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

    // 找到当前block处理数组段中的位置
    int *idata = g_idata + blockIdx.x * blockDim.x * 8;
    
    if(idx+7*blockDim.x < n){
        int a = g_idata[idx];
        int b = g_idata[idx+blockDim.x];
        int c = g_idata[idx+2*blockDim.x];
        int d = g_idata[idx+3*blockDim.x];
        int e = g_idata[idx+4*blockDim.x];
        int f = g_idata[idx+5*blockDim.x];
        int g = g_idata[idx+6*blockDim.x];
        int h = g_idata[idx+7*blockDim.x];
        g_idata[idx] = a + b + c + d + e + f + g + h;
    } 
    __syncthreads();
    for(int stride = blockDim.x/2; stride > 32; stride >>= 1){
        if(tid < stride){
            idata[tid] += idata[tid+stride];
        }
        __syncthreads();
    }
    if(tid < 32){
        volatile int *vmem = idata;
        vmem[tid] += vmem[tid+32];
        vmem[tid] += vmem[tid+16];
        vmem[tid] += vmem[tid+8];
        vmem[tid] += vmem[tid+4];
        vmem[tid] += vmem[tid+2];
        vmem[tid] += vmem[tid+1];
    }
    // 该数组段，即该block需要处理的数组段 相加的结果 放在idata[0], 然后将结果放到g_odata对应的block.x的位置
    if(tid == 0) g_odata[blockIdx.x] = idata[0];
}

template<unsigned int iBlockSize>
__global__ void reduceCompleteUnroll(int *g_idata, int *g_odata, unsigned int n){

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

    // 找到当前block处理数组段中的位置
    int *idata = g_idata + blockIdx.x * blockDim.x * 8;
    
    if(idx+7*blockDim.x < n){
        int a = g_idata[idx];
        int b = g_idata[idx+blockDim.x];
        int c = g_idata[idx+2*blockDim.x];
        int d = g_idata[idx+3*blockDim.x];
        int e = g_idata[idx+4*blockDim.x];
        int f = g_idata[idx+5*blockDim.x];
        int g = g_idata[idx+6*blockDim.x];
        int h = g_idata[idx+7*blockDim.x];
        g_idata[idx] = a + b + c + d + e + f + g + h;
    } 
    __syncthreads();
    if(iBlockSize >=1024 && tid <512) idata[tid] += idata[tid + 512];
    __syncthreads();
    if(iBlockSize >=512 && tid <256) idata[tid] += idata[tid + 256];
    __syncthreads();
    if(iBlockSize >=256 && tid <128) idata[tid] += idata[tid + 128];
    __syncthreads();
    if(iBlockSize >=128 && tid <64) idata[tid] += idata[tid + 64];
    __syncthreads();
    if(tid < 32){
        volatile int *vmem = idata;
        vmem[tid] += vmem[tid+32];
        vmem[tid] += vmem[tid+16];
        vmem[tid] += vmem[tid+8];
        vmem[tid] += vmem[tid+4];
        vmem[tid] += vmem[tid+2];
        vmem[tid] += vmem[tid+1];
    }
    // 该数组段，即该block需要处理的数组段 相加的结果 放在idata[0], 然后将结果放到g_odata对应的block.x的位置
    if(tid == 0) g_odata[blockIdx.x] = idata[0];
}


int main(int argc, char** argv){

    // printf("%s Starting...\n", argv[0]);
    int dev = 0;
    double iStart, iElaps;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    // printf("Use Device: %d %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));


    int size = 1<<24;
    size_t nbytes = size * sizeof(int);
    int *h_idata, *tmp;
    h_idata =(int *)malloc(size * sizeof(int));
    tmp =(int *)malloc(size * sizeof(int));

    for(int i =0; i < size; i++){
        h_idata[i] = (int)( rand() & 0xFF );
    }

    memcpy(tmp, h_idata, size*sizeof(int));
    iStart = cpuSecond();
    int cpu_sum = recursiveReduce(tmp, size);
    iElaps = cpuSecond() - iStart;
    printf("cpu reduce      elapsed %f sec cpu_sum: %d\n", iElaps, cpu_sum);

    int block_size = 1024;
    if(argc > 1){
        block_size = atoi(argv[1]);
    }
    dim3 block(block_size, 1);
    dim3 grid((size+block.x - 1)/block.x , 1);

    int *g_idata, *g_odata;
    cudaMalloc((void**)&g_idata, size*sizeof(int));
    cudaMalloc((void**)&g_odata, grid.x*sizeof(int));

    int* h_odata = (int*)malloc(grid.x * sizeof(int));

    CHECK(cudaMemcpy(g_idata, h_idata, nbytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = cpuSecond();
    reduceNeighbored<<<grid, block>>>(g_idata, g_odata, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = cpuSecond() - iStart;
    cudaMemcpy(h_odata, g_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost);

    int gpu_sum = 0;
    for(int i =0; i<grid.x; i++) gpu_sum+=h_odata[i];

    printf("gpu Neighbored  elapsed %f sec gpu_sum: %d <<<grid %d block "
           "%d>>>\n", iElaps, gpu_sum, grid.x, block.x);
    
    CHECK(cudaMemcpy(g_idata, h_idata, nbytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = cpuSecond();
    reduceNeighboredLess<<<grid, block>>>(g_idata, g_odata, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = cpuSecond() - iStart;
    CHECK(cudaMemcpy(h_odata, g_odata, grid.x * sizeof(int),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];

    printf("gpu NeighboredLess elapsed %f sec gpu_sum: %d <<<grid %d block "
           "%d>>>\n", iElaps, gpu_sum, grid.x, block.x);

    CHECK(cudaMemcpy(g_idata, h_idata, nbytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = cpuSecond();
    reduceNeighboredLeaved<<<grid, block>>>(g_idata, g_odata, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = cpuSecond() - iStart;
    CHECK(cudaMemcpy(h_odata, g_odata, grid.x * sizeof(int),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];

    printf("gpu NeighboredLeved elapsed %f sec gpu_sum: %d <<<grid %d block "
           "%d>>>\n", iElaps, gpu_sum, grid.x, block.x);

    CHECK(cudaMemcpy(g_idata, h_idata, nbytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = cpuSecond();
    reduceUnrolling2<<<grid.x/2, block>>>(g_idata, g_odata, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = cpuSecond() - iStart;
    CHECK(cudaMemcpy(h_odata, g_odata, grid.x * sizeof(int),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x/2; i++) gpu_sum += h_odata[i];

    printf("gpu reduceUnrolling2 elapsed %f sec gpu_sum: %d <<<grid %d block "
           "%d>>>\n", iElaps, gpu_sum, grid.x, block.x);


    CHECK(cudaMemcpy(g_idata, h_idata, nbytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = cpuSecond();
    reduceUnrolling4<<<grid.x/4, block>>>(g_idata, g_odata, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = cpuSecond() - iStart;
    CHECK(cudaMemcpy(h_odata, g_odata, grid.x * sizeof(int),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x/4; i++) gpu_sum += h_odata[i];

    printf("gpu reduceUnrolling4 elapsed %f sec gpu_sum: %d <<<grid %d block "
           "%d>>>\n", iElaps, gpu_sum, grid.x, block.x);

    CHECK(cudaMemcpy(g_idata, h_idata, nbytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = cpuSecond();
    reduceUnrolling8<<<grid.x/8, block>>>(g_idata, g_odata, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = cpuSecond() - iStart;
    CHECK(cudaMemcpy(h_odata, g_odata, grid.x * sizeof(int),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x/8; i++) gpu_sum += h_odata[i];

    printf("gpu reduceUnrolling8 elapsed %f sec gpu_sum: %d <<<grid %d block "
           "%d>>>\n", iElaps, gpu_sum, grid.x, block.x);


    CHECK(cudaMemcpy(g_idata, h_idata, nbytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = cpuSecond();
    reduceUnrollWarps8<<<grid.x/8, block>>>(g_idata, g_odata, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = cpuSecond() - iStart;
    CHECK(cudaMemcpy(h_odata, g_odata, grid.x * sizeof(int),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x/8; i++) gpu_sum += h_odata[i];

    printf("gpu reduceUnrollWarps8 elapsed %f sec gpu_sum: %d <<<grid %d block "
           "%d>>>\n", iElaps, gpu_sum, grid.x, block.x);
    
    CHECK(cudaMemcpy(g_idata, h_idata, nbytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = cpuSecond();
    reduceCompleteUnrollWarps8<<<grid.x/8, block>>>(g_idata, g_odata, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = cpuSecond() - iStart;
    CHECK(cudaMemcpy(h_odata, g_odata, grid.x * sizeof(int),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x/8; i++) gpu_sum += h_odata[i];

    printf("gpu reduceCompleteUnrollWarps8 elapsed %f sec gpu_sum: %d <<<grid %d block "
           "%d>>>\n", iElaps, gpu_sum, grid.x, block.x);
    
    CHECK(cudaMemcpy(g_idata, h_idata, nbytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = cpuSecond();
    switch(block_size){
        case 1024:
            reduceCompleteUnroll<1024><<<grid.x/8, block>>>(g_idata, g_odata, size);
            break;
        case 512:
            reduceCompleteUnroll<512><<<grid.x/8, block>>>(g_idata, g_odata, size);
            break;
        case 256:
            reduceCompleteUnroll<256><<<grid.x/8, block>>>(g_idata, g_odata, size);
            break;
        case 128:
            reduceCompleteUnroll<128><<<grid.x/8, block>>>(g_idata, g_odata, size);
            break;
        case 64:
            reduceCompleteUnroll<64><<<grid.x/8, block>>>(g_idata, g_odata, size);
            break;
    }
    CHECK(cudaDeviceSynchronize());
    iElaps = cpuSecond() - iStart;
    CHECK(cudaMemcpy(h_odata, g_odata, grid.x * sizeof(int),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x/8; i++) gpu_sum += h_odata[i];

    printf("gpu reduceCompleteUnroll<%d> elapsed %f sec gpu_sum: %d <<<grid %d block "
           "%d>>>\n", block_size, iElaps, gpu_sum, grid.x, block.x);
    

    // free host memory
    free(h_idata);
    free(h_odata);

    // free device memory
    CHECK(cudaFree(g_idata));
    CHECK(cudaFree(g_odata));

    // reset device
    CHECK(cudaDeviceReset());
    return 0;
}