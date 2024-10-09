#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#if defined _WIN32
    #include "win.h"
#else
    #include <unistd.h>
    #include <sys/mman.h>
#endif

void read_datafile(char* file_name, void* data, size_t offset, size_t size) {
    FILE *file = fopen(file_name, "rb");

    fseek(file, offset, SEEK_SET);

    fread(data, size, 1, file);

    fclose(file);
}

void quant_absmax(float* x, signed char* q_x, float* s_x) {
    int size = 512;
    float abs_max = fabs(x[0]);
    for(int i=1; i<size; i++) {
        abs_max = fmax(abs_max, fabs(x[i]));
    }

    *s_x = 127 / abs_max;

    for(int i=0; i<size; i++) {
        q_x[i] = (signed char)roundf(x[i] * (*s_x));
    }
}

void matmul_w4a8(float* result, signed char* q_x, unsigned char* weight, int M, int N, int K, float* alpha, float s_x) {
    int mm_int[M][N];

    for(int m=0; m<M; m++) {
        for(int n=0; n<N; n++) {
            int sum = 0;
            for(int k=0; k<K; k++) {
                unsigned char w = weight[m*K+k];
                sum += (w / 16) * q_x[n*K*2 + 2*k] + (w % 16) * q_x[n*K*2 + 2*k+1];
            }
            //printf("%d ", sum);
            mm_int[m][n] = sum;
        }
    }

    float sum_q_x = 0;

    for(int i=0; i<K*2; i++) {
        sum_q_x += (float)q_x[i];
    }

    for(int i=0; i<M; i++) {
        result[i] = (mm_int[i][0] * alpha[i*2+1] + alpha[i*2] * sum_q_x) / s_x;
    }
}

int main(int argc, char *argv[]) {
    float* x = (float*)malloc(512*sizeof(float));
    signed char* q_x = (signed char*)malloc(512*sizeof(signed char));
    float s_x;
    float* result = (float*)malloc(512*sizeof(float));
    float* golden = (float*)malloc(512*sizeof(float));

    unsigned char* weight = (unsigned char*)malloc(256*512*sizeof(unsigned char));
    float* alpha = (float*)malloc(1024*sizeof(float));

    read_datafile("x.bin", x, 0, 512*sizeof(float));
    read_datafile("golden.bin", golden, 0, 512*sizeof(float));

    read_datafile("weight.bin", weight, 0, 256*512*sizeof(unsigned char));
    read_datafile("weight.bin", alpha, 131072, 1024*sizeof(float));

    quant_absmax(x, q_x, &s_x);

    matmul_w4a8(result, q_x, weight, 512, 1, 256, alpha, s_x);

    float diff_max = 0;
    float diff_min = 0;

    for(int i=0; i<512; i++) {
        diff_max = fmax(diff_max, result[i]-golden[i]);
        diff_min = fmin(diff_min, result[i]-golden[i]);
    }

    printf("%.5f %.5f\n", diff_max, diff_min);

    free(x);
    free(q_x);
    free(result);
    free(golden);
    free(weight);
    free(alpha);

    return 0;
}