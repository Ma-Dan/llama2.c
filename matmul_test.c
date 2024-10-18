#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <stdint.h>
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

// ----------------------------------------------------------------------------
// Globals
int GS = 0; // group size global for quantization of the weights

typedef struct {
    int8_t* q;    // quantized values
    float* s; // scaling factors
} QuantizedTensor;

// ----------------------------------------------------------------------------
// Quantization functions

void dequantize(QuantizedTensor *qx, float* x, int n) {
    for (int i = 0; i < n; i++) {
        x[i] = qx->q[i] * qx->s[i / GS];
    }
}

void quantize(QuantizedTensor *qx, float* x, int n) {
    int num_groups = n / GS;
    float Q_MAX = 127.0f;

    for (int group = 0; group < num_groups; group++) {

        // find the max absolute value in the current group
        float wmax = 0.0;
        for (int i = 0; i < GS; i++) {
            float val = fabs(x[group * GS + i]);
            if (val > wmax) {
                wmax = val;
            }
        }

        // calculate and write the scaling factor
        float scale = wmax / Q_MAX;
        qx->s[group] = scale;

        // calculate and write the quantized values
        for (int i = 0; i < GS; i++) {
            float quant_value = x[group * GS + i] / scale; // scale
            int8_t quantized = (int8_t) round(quant_value); // round and clamp
            qx->q[group * GS + i] = quantized;
        }
    }
}

void matmul(float* xout, QuantizedTensor *x, QuantizedTensor *w, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    // inputs to this function are both quantized

    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < d; i++) {

        float val = 0.0f;
        int32_t ival = 0;
        int in = i * n;

        // do the matmul in groups of GS
        int j;
        for (j = 0; j <= n - GS; j += GS) {
            for (int k = 0; k < GS; k++) {
                ival += ((int32_t) x->q[j + k]) * ((int32_t) w->q[in + j + k]);
            }
            val += ((float) ival) * w->s[(in + j) / GS] * x->s[j / GS];
            ival = 0;
        }

        xout[i] = val;
    }
}

void matmul_float(float* xout, float* x, float* w, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}

void read_datafile(char* file_name, void* data, size_t offset, size_t size) {
    FILE *file = fopen(file_name, "rb");

    fseek(file, offset, SEEK_SET);

    fread(data, size, 1, file);

    fclose(file);
}

/* initialize `n` x quantized tensor (with `size_each` elements), starting from memory pointed at *ptr */
QuantizedTensor *init_quantized_tensors(void **ptr, int n, int size_each) {
    void *p = *ptr;
    QuantizedTensor *res = (QuantizedTensor*)malloc(n * sizeof(QuantizedTensor));
    for(int i=0; i<n; i++) {
        /* map quantized int8 values*/
        res[i].q = (int8_t*)p;
        p = (int8_t*)p + size_each;
        /* map scale factors */
        res[i].s = (float*)p;
        p = (float*)p + size_each / GS;
    }
    *ptr = p; // advance ptr to current position
    return res;
}

int main(int argc, char** argv) {
    int dim = 512;

    float* x = (float*)malloc(dim*sizeof(float));
    float* result = (float*)malloc(dim*sizeof(float));

    float* golden = (float*)malloc(dim*sizeof(float));
    float* weight = (float*)malloc(dim*dim*sizeof(float));

    read_datafile("x.bin", x, 0, dim*sizeof(float));
    read_datafile("golden.bin", golden, 0, dim*sizeof(float));
    read_datafile("weight.bin", weight, 0, dim*dim*sizeof(float));

    GS = 64;
    FILE *file_weight_q = fopen("weight_q.bin", "rb");
    fseek(file_weight_q, 0, SEEK_END);
    size_t weight_q_len = ftell(file_weight_q);
    unsigned char* weight_q = (unsigned char*)malloc(weight_q_len);
    fseek(file_weight_q, 0, SEEK_SET);
    fread(weight_q, 1, weight_q_len, file_weight_q);
    fclose(file_weight_q);
    QuantizedTensor* wq = init_quantized_tensors(&weight_q, 1, dim*dim);

    // float matmul
    matmul_float(result, x, weight, dim, dim);

    float diff_max = 0;
    float diff_min = 0;
    for(int i=0; i<dim; i++) {
        diff_max = fmax(diff_max, result[i]-golden[i]);
        diff_min = fmin(diff_min, result[i]-golden[i]);
    }
    printf("%.5f %.5f\n", diff_max, diff_min);


    // W8A8 matmul
    QuantizedTensor xq = (QuantizedTensor) { .q = (int8_t*)calloc(dim, sizeof(int8_t)), .s = (float*)calloc(dim, sizeof(float)) };
    quantize(&xq, x, dim);
    matmul(result, &xq, wq, dim, dim);

    diff_max = 0;
    diff_min = 0;
    for(int i=0; i<dim; i++) {
        diff_max = fmax(diff_max, result[i]-golden[i]);
        diff_min = fmin(diff_min, result[i]-golden[i]);
    }
    printf("%.5f %.5f\n", diff_max, diff_min);

    free(x);
    free(result);
    free(golden);
    free(weight);
    free(wq->q);
    free(wq);
    free(xq.q);
    free(xq.s);

    return 0;
}