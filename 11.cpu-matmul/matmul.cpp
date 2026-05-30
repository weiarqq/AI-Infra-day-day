#include <i386/types.h>
#include <iostream>
#include <vector>

#include <sys/time.h>

double cpuSecond()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);

    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

void print_mat(float* C, int M, int N)
{
    std::cout << "[" << std::endl;
    for (int m = 0; m < M; m++) {
        std::cout << "[";
        for (int n = 0; n < N; n++) {
            std::cout << C[m * N + n] << ", ";
        }
        std::cout << "]" << std::endl;
    }
    std::cout << "]" << std::endl;
}

void native_hgemm(float* A, float* B, float* C, int M, int N, int K)
{
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            for (int k = 0; k < K; k++) {
                C[m * N + n] = C[m * N + n] + A[m * K + k] * B[k * N + n];
            }
        }
    }
}
void native_hgemm_v1(float* A, float* B, float* C, int M, int N, int K)
{
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float temp = 0.0f;
            for (int k = 0; k < K; k++) {
                temp += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] = temp;
        }
    }
}
void native_hgemm_v2(float* A, float* B, float* C, int M, int N, int K)
{
    for (int n = 0; n < N; n++) {
        for (int m = 0; m < M; m++) {
            float temp = 0.0f;
            for (int k = 0; k < K; k++) {
                temp += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] = temp;
        }
    }
}

// unroll 一次计算4个元素
void hgemm_roll(float* A, float* B, float* C, int M, int N, int K)
{
    for (int n = 0; n < N; n += 4) {
        for (int m = 0; m < M; m++) {
            float temp[4] = { 0.0f };
            float a_val = 0.0f;
            for (int k = 0; k < K; k++) {
                a_val = A[m * K + k]; // 一次加载，四次使用
                temp[0] += a_val * B[k * N + n + 0];
                temp[1] += a_val * B[k * N + n + 1];
                temp[2] += a_val * B[k * N + n + 2];
                temp[3] += a_val * B[k * N + n + 3];
            }
            C[m * N + n + 0] = temp[0];
            C[m * N + n + 1] = temp[1];
            C[m * N + n + 2] = temp[2];
            C[m * N + n + 3] = temp[3];
            // #pragma unroll
            //             for (int t = 0; t < 4; t++) {
            //                 C[m * N + n + t] = temp[t];
            //             }
        }
    }
}

// unroll 一次计算4个元素
void hgemm_roll_kn(float* A, float* B, float* C, int M, int N, int K)
{
    for (int n = 0; n < N; n += 4) {
        for (int m = 0; m < M; m++) {
            float temp[4] = { 0.0f };
            float a_val[4] = { 0.0f };
            for (int k = 0; k < K; k += 4) {
                a_val[0] = A[m * K + k]; // 一次加载，四次使用
                temp[0] += a_val[0] * B[k * N + n + 0];
                temp[1] += a_val[0] * B[k * N + n + 1];
                temp[2] += a_val[0] * B[k * N + n + 2];
                temp[3] += a_val[0] * B[k * N + n + 3];

                a_val[1] = A[m * K + k + 1]; // 一次加载，四次使用
                temp[0] += a_val[1] * B[k * N + n + 0];
                temp[1] += a_val[1] * B[k * N + n + 1];
                temp[2] += a_val[1] * B[k * N + n + 2];
                temp[3] += a_val[1] * B[k * N + n + 3];

                a_val[2] = A[m * K + k + 2]; // 一次加载，四次使用
                temp[0] += a_val[2] * B[k * N + n + 0];
                temp[1] += a_val[2] * B[k * N + n + 1];
                temp[2] += a_val[2] * B[k * N + n + 2];
                temp[3] += a_val[2] * B[k * N + n + 3];

                a_val[3] = A[m * K + k + 3]; // 一次加载，四次使用
                temp[0] += a_val[3] * B[k * N + n + 0];
                temp[1] += a_val[3] * B[k * N + n + 1];
                temp[2] += a_val[3] * B[k * N + n + 2];
                temp[3] += a_val[3] * B[k * N + n + 3];
            }
            C[m * N + n + 0] = temp[0];
            C[m * N + n + 1] = temp[1];
            C[m * N + n + 2] = temp[2];
            C[m * N + n + 3] = temp[3];
        }
    }
}

void hgemm_roll_knp(float* A, float* B, float* C, int M, int N, int K)
{
    for (int n = 0; n < N; n += 4) {
        for (int m = 0; m < M; m++) {
            float temp[4] = { 0.0f };
            float a_val[4] = { 0.0f };
            float* A_p = A + m * K;

            for (int k = 0; k < K; k += 4) {
                a_val[0] = *(A_p + k); // 一次加载，四次使用
                a_val[1] = *(A_p + k + 1); // 一次加载，四次使用
                a_val[2] = *(A_p + k + 2); // 一次加载，四次使用
                a_val[3] = *(A_p + k + 3); // 一次加载，四次使用
                int b_i = k * N + n;

                temp[0] += a_val[0] * B[b_i];
                temp[1] += a_val[0] * B[b_i + 1];
                temp[2] += a_val[0] * B[b_i + 2];
                temp[3] += a_val[0] * B[b_i + 3];

                temp[0] += a_val[1] * B[b_i];
                temp[1] += a_val[1] * B[b_i + 1];
                temp[2] += a_val[1] * B[b_i + 2];
                temp[3] += a_val[1] * B[b_i + 3];

                temp[0] += a_val[2] * B[b_i];
                temp[1] += a_val[2] * B[b_i + 1];
                temp[2] += a_val[2] * B[b_i + 2];
                temp[3] += a_val[2] * B[b_i + 3];

                temp[0] += a_val[3] * B[b_i];
                temp[1] += a_val[3] * B[b_i + 1];
                temp[2] += a_val[3] * B[b_i + 2];
                temp[3] += a_val[3] * B[b_i + 3];
            }
            int index = m * N + n;
            C[index + 0] = temp[0];
            C[index + 1] = temp[1];
            C[index + 2] = temp[2];
            C[index + 3] = temp[3];
        }
    }
}

void hgemm_roll_knp2(float* A, float* B, float* C, int M, int N, int K)
{
    for (int n = 0; n < N; n += 4) {
        for (int m = 0; m < M; m++) {
            float temp[4] = { 0.0f };
            float* A_p = A + m * K;

            for (int k = 0; k < K; k += 4) {
                int b_i = k * N + n;

                temp[0] += *(A_p + k) * B[b_i];
                temp[1] += *(A_p + k) * B[b_i + 1];
                temp[2] += *(A_p + k) * B[b_i + 2];
                temp[3] += *(A_p + k) * B[b_i + 3];

                temp[0] += *(A_p + k + 1) * B[b_i];
                temp[1] += *(A_p + k + 1) * B[b_i + 1];
                temp[2] += *(A_p + k + 1) * B[b_i + 2];
                temp[3] += *(A_p + k + 1) * B[b_i + 3];

                temp[0] += *(A_p + k + 2) * B[b_i];
                temp[1] += *(A_p + k + 2) * B[b_i + 1];
                temp[2] += *(A_p + k + 2) * B[b_i + 2];
                temp[3] += *(A_p + k + 2) * B[b_i + 3];

                temp[0] += *(A_p + k + 3) * B[b_i];
                temp[1] += *(A_p + k + 3) * B[b_i + 1];
                temp[2] += *(A_p + k + 3) * B[b_i + 2];
                temp[3] += *(A_p + k + 3) * B[b_i + 3];
            }
            int index = m * N + n;
            C[index + 0] = temp[0];
            C[index + 1] = temp[1];
            C[index + 2] = temp[2];
            C[index + 3] = temp[3];
        }
    }
}

void hgemm_roll_m4n4k(float* __restrict A, float* __restrict B, float* __restrict C, int M, int N, int K)
{
    for (int m = 0; m < M; m += 4) {
        for (int n = 0; n < N; n += 4) {

            float t00 = 0, t01 = 0, t02 = 0, t03 = 0;
            float t10 = 0, t11 = 0, t12 = 0, t13 = 0;
            float t20 = 0, t21 = 0, t22 = 0, t23 = 0;
            float t30 = 0, t31 = 0, t32 = 0, t33 = 0;

            float* A0 = A + (m + 0) * K;
            float* A1 = A + (m + 1) * K;
            float* A2 = A + (m + 2) * K;
            float* A3 = A + (m + 3) * K;

            for (int k = 0; k < K; k++) {

                // 👉 正确使用 k
                float a0 = A0[k];
                float a1 = A1[k];
                float a2 = A2[k];
                float a3 = A3[k];

                // 👉 B 连续访问
                float* Bp = B + k * N + n;
                float b0 = Bp[0];
                float b1 = Bp[1];
                float b2 = Bp[2];
                float b3 = Bp[3];

                // 👉 16 FMA（编译器容易优化）
                t00 += a0 * b0;
                t10 += a1 * b0;
                t20 += a2 * b0;
                t30 += a3 * b0;

                t01 += a0 * b1;
                t11 += a1 * b1;
                t21 += a2 * b1;
                t31 += a3 * b1;

                t02 += a0 * b2;
                t12 += a1 * b2;
                t22 += a2 * b2;
                t32 += a3 * b2;

                t03 += a0 * b3;
                t13 += a1 * b3;
                t23 += a2 * b3;
                t33 += a3 * b3;
            }

            C[(m + 0) * N + n + 0] = t00;
            C[(m + 0) * N + n + 1] = t01;
            C[(m + 0) * N + n + 2] = t02;
            C[(m + 0) * N + n + 3] = t03;

            C[(m + 1) * N + n + 0] = t10;
            C[(m + 1) * N + n + 1] = t11;
            C[(m + 1) * N + n + 2] = t12;
            C[(m + 1) * N + n + 3] = t13;

            C[(m + 2) * N + n + 0] = t20;
            C[(m + 2) * N + n + 1] = t21;
            C[(m + 2) * N + n + 2] = t22;
            C[(m + 2) * N + n + 3] = t23;

            C[(m + 3) * N + n + 0] = t30;
            C[(m + 3) * N + n + 1] = t31;
            C[(m + 3) * N + n + 2] = t32;
            C[(m + 3) * N + n + 3] = t33;
        }
    }
}

void hgemm_roll_m4n4k2(float* __restrict A, float* __restrict B, float* __restrict C, int M, int N, int K)
{
    for (int m = 0; m < M; m += 2) {
        for (int n = 0; n < N; n += 2) {

            float t00 = 0, t01 = 0;
            float t10 = 0, t11 = 0;
            // float t00 = 0, t01 = 0, t02 = 0, t03 = 0;
            // float t10 = 0, t11 = 0, t12 = 0, t13 = 0;
            // float t20 = 0, t21 = 0, t22 = 0, t23 = 0;
            // float t30 = 0, t31 = 0, t32 = 0, t33 = 0;

            float* A0 = A + (m + 0) * K;
            float* A1 = A + (m + 1) * K;
            // float* A2 = A + (m + 2) * K;
            // float* A3 = A + (m + 3) * K;

            for (int k = 0; k < K; k++) {

                // 👉 正确使用 k
                float a0 = A0[k];
                float a1 = A1[k];
                // float a2 = A2[k];
                // float a3 = A3[k];

                // 👉 B 连续访问
                float* Bp = B + k * N + n;
                float b0 = Bp[0];
                float b1 = Bp[1];
                // float b2 = Bp[2];
                // float b3 = Bp[3];

                t00 += a0 * b0;
                t10 += a1 * b0;
                // t20 += a2 * b0;
                // t30 += a3 * b0;

                t01 += a0 * b1;
                t11 += a1 * b1;
                // t21 += a2 * b1;
                // t31 += a3 * b1;

                // t02 += a0 * b2;
                // t12 += a1 * b2;
                // t22 += a2 * b2;
                // t32 += a3 * b2;

                // t03 += a0 * b3;
                // t13 += a1 * b3;
                // t23 += a2 * b3;
                // t33 += a3 * b3;
            }

            C[(m + 0) * N + n + 0] = t00;
            C[(m + 0) * N + n + 1] = t01;
            // C[(m + 0) * N + n + 2] = t02;
            // C[(m + 0) * N + n + 3] = t03;

            C[(m + 1) * N + n + 0] = t10;
            C[(m + 1) * N + n + 1] = t11;
            // C[(m + 1) * N + n + 2] = t12;
            // C[(m + 1) * N + n + 3] = t13;

            // C[(m + 2) * N + n + 0] = t20;
            // C[(m + 2) * N + n + 1] = t21;
            // C[(m + 2) * N + n + 2] = t22;
            // C[(m + 2) * N + n + 3] = t23;

            // C[(m + 3) * N + n + 0] = t30;
            // C[(m + 3) * N + n + 1] = t31;
            // C[(m + 3) * N + n + 2] = t32;
            // C[(m + 3) * N + n + 3] = t33;
        }
    }
}

int main()
{
    double iStart, iElaps;
    const int m = 128;
    const int n = 2048;
    const int k = 2048;

    std::vector<float> A(m * k, 1.0f);
    std::vector<float> B(n * k, 1.0f);

    std::vector<float> C(m * n, 0.0f);

    // print_mat(C.data(), 10, 10);

    // iStart = cpuSecond();
    // native_hgemm(A.data(), B.data(), C.data(), m, n, k);
    // printf("native_hgemm Time elaspsed %f sec \n", cpuSecond() - iStart);
    // iStart = cpuSecond();
    // native_hgemm_v1(A.data(), B.data(), C.data(), m, n, k);
    // printf("native_hgemm_v1 Time elaspsed %f sec \n", cpuSecond() - iStart);
    iStart = cpuSecond();
    native_hgemm_v2(A.data(), B.data(), C.data(), m, n, k);
    printf("native_hgemm_v2 Time elaspsed %f sec \n", cpuSecond() - iStart);

    iStart = cpuSecond();
    hgemm_roll(A.data(), B.data(), C.data(), m, n, k);
    printf("hgemm_roll Time elaspsed %f sec \n", cpuSecond() - iStart);

    iStart = cpuSecond();
    hgemm_roll_kn(A.data(), B.data(), C.data(), m, n, k);
    printf("hgemm_roll_kn Time elaspsed %f sec \n", cpuSecond() - iStart);

    iStart = cpuSecond();
    hgemm_roll_knp(A.data(), B.data(), C.data(), m, n, k);
    printf("hgemm_roll_knp Time elaspsed %f sec \n", cpuSecond() - iStart);

    iStart = cpuSecond();
    hgemm_roll_knp2(A.data(), B.data(), C.data(), m, n, k);
    printf("hgemm_roll_knp2 Time elaspsed %f sec \n", cpuSecond() - iStart);

    iStart = cpuSecond();
    hgemm_roll_m4n4k(A.data(), B.data(), C.data(), m, n, k);
    printf("hgemm_roll_m4n4k Time elaspsed %f sec \n", cpuSecond() - iStart);

    print_mat(C.data(), 10, 10);
}