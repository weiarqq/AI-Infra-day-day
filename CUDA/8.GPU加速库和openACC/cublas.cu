#include "common.h"
#include "cublas_v2.h"
#include <cstdlib>
#include <cuda.h>

int M = 1024;
int N = 1024;

void generate_random_vector(int N, float** outX)
{
    float* X = new float[N];
    for (int i = 0; i < N; i++) {
        X[i] = (double)rand() / (double)RAND_MAX;
    }

    *outX = X;
}

void generate_random_dense_matrix(int M, int N, float** outA)
{
    float* A = new float[M * N];
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < M; i++) {
            A[j * M + i] = ((double)rand() / (double)RAND_MAX) * 100.0;
        }
    }
    *outA = A;
}

int main(int argc, char** argv)
{
    int i;
    float *A, *dA;
    float *X, *dX;
    float *Y, *dY;
    float beta;
    float alpha;

    alpha = 3.0f;
    beta = 4.0f;

    // Generate inputs
    srand(9384);
    generate_random_dense_matrix(M, N, &A);
    generate_random_vector(N, &X);
    generate_random_vector(M, &Y);

    cublasHandle_t handle = 0;
    CHECK_CUBLAS(cublasCreate(&handle));
    CHECK(cudaMalloc((float**)&dA, sizeof(float) * M * N));
    CHECK(cudaMalloc((void**)&dX, sizeof(float) * N));
    CHECK(cudaMalloc((void**)&dY, sizeof(float) * M));
    CHECK_CUBLAS(cublasSetMatrix(M, N, sizeof(float), A, M, dA, N));
    CHECK_CUBLAS(cublasSetVector(M, sizeof(float), Y, 1, dY, 1));
    CHECK_CUBLAS(cublasSetVector(N, sizeof(float), X, 1, dX, 1));

    CHECK_CUBLAS(cublasSgemv_v2(handle, CUBLAS_OP_N, M, N, &alpha, dA, M, dX, 1, &beta, dY, 1));

    CHECK_CUBLAS(cublasGetVector(M, sizeof(float), dY, 1, Y, 1));

    for (i = 0; i < 10; i++) {
        printf("%2.2f\n", Y[i]);
    }

    printf("...\n");

    free(A);
    free(X);
    free(Y);

    CHECK(cudaFree(dA));
    CHECK(cudaFree(dY));
    CHECK_CUBLAS(cublasDestroy(handle));

    return 0;
}
