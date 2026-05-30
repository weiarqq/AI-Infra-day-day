#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>



void sumArrayOnHost(float* A, float* B, float* C, const int N){
    for(int i=0; i< N; i++){
        C[i] = A[i] + B[i];
    }
}

void initialData(float* ip, int size){
    time_t t;
    srand((unsigned int) time(&t));
    for(int i=0; i<size; i++){
        ip[i] = (float)(rand() & 0xFF)/10.0f;
    }
}


int main(int argc, char** argv){
    int nElem = 1024;
    float *h_A, *h_B, *h_C;
    size_t nBytes = nElem * sizeof(float);
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    h_C = (float *)malloc(nBytes);

    initialData(h_A, nElem);
    initialData(h_B, nElem);

    sumArrayOnHost(h_A, h_B, h_C, nElem);
    for(int i=0; i< nElem; i++){
        printf("%f ", h_C[i]); // 每个值后加空格分隔
    }

    free(h_A);
    free(h_B);
    free(h_C);
    return 0;
}