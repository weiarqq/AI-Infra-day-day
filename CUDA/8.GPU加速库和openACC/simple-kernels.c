#include <stdio.h>
#include <stdlib.h>

#define N 1024

int main(int argc, char** argv)
{
    int i;

    int* __restrict__ A = (int*)malloc(N * sizeof(int));
    int* __restrict__ B = (int*)malloc(N * sizeof(int));
    int* __restrict__ C = (int*)malloc(N * sizeof(int));
    int* __restrict__ D = (int*)malloc(N * sizeof(int));

    for (i = 0; i < N; i++) {
        A[i] = i;
        B[i] = 2 * i;
    }

#pragma acc kernels
    for (i = 0; i < N; i++)
        C[i] = A[i] + B[i];
    for (i = 0; i < N; i++)
        D[i] = C[i] * A[i];

    for (i = 0; i < 10; i++) {
        printf("%d ", D[i]);
    }

    printf("...\n");

    return 0;
}
