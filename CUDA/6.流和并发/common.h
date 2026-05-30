#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#ifndef MY_MACROS_H // 头文件保护（防止重复包含）
#define MY_MACROS_H

#define CHECK(call)                                                             \
    {                                                                           \
        const cudaError_t error = call;                                         \
        if (error != cudaSuccess) {                                             \
            printf("Error: %s:%d", __FILE__, __LINE__);                         \
            printf("code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
            exit(1);                                                            \
        }                                                                       \
    }
#endif

void checkResult(float* hostRef, float* gpuRef, const int N);

double cpuSecond();

void initialData(float* ip, int size);

void checkResultOffset(float* hostRef, float* gpuRef, const int N, const int offset);

void printData(float* in, const int size);