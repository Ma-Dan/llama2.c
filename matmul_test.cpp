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
#include <sys/time.h>

#define NEON

#ifdef NEON
#include <arm_neon.h>
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
        //printf("matmul %.5f\n", val);
    }
}

#ifdef NEON

inline int32x4_t muladd(const int8x16_t & a, const int8x16_t & b)
{
    int16x8_t lo = vmull_s8(vget_low_s8(a), vget_low_s8(b));
    int16x8_t hi = vmull_s8(vget_high_s8(a), vget_high_s8(b));
    return vaddq_s32(vpaddlq_s16(lo), vpaddlq_s16(hi));
}

void gemv_neon(float* xout, QuantizedTensor *x, QuantizedTensor *w, int n, int d) {
    int8x16_t column;
    int8x16_t element;

    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < d; i++) {
        float sum = 0.0f;
        int in = i * n;
        for(int j = 0; j <= n - GS; j += GS) {
            int32x4_t sum_gs = vdupq_n_s32(0);
            for(int k = 0; k < GS; k += 16) {
                column = vld1q_s8(w->q + in + j + k);
                element = vld1q_s8(x->q + j + k);
                sum_gs += muladd(column, element);
            }
            int32x2_t sum2 = vpadd_s32(vget_high_s32(sum_gs), vget_low_s32(sum_gs));
            int rsum = vget_lane_s32(sum2, 0) + vget_lane_s32(sum2, 1);
            sum += ((float) rsum) * w->s[(in + j) / GS] * x->s[j / GS];
        }

        xout[i] = sum;
        //printf("matmuq %.5f\n", sum);
    }
}

#endif

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

#ifdef NEON

void gemv_neon_float(float* result, float* x, float* w, int n, int d) {
    float32x4_t column;
    float32x4_t element;

    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < d; i++) {
        float32x4_t sum = vdupq_n_f32(0.0f);
        for(int j=0; j < n; j+=4) {
            column = vld1q_f32(w + i * n + j);
            element = vld1q_f32(x + j);
            sum = vmlaq_f32(sum, column, element);
        }
        float32x2_t sum2 = vpadd_f32(vget_high_f32(sum), vget_low_f32(sum));
        float rsum = vget_lane_f32(sum2, 0) + vget_lane_f32(sum2, 1);
        result[i] = rsum;
    }
}

#endif

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
    QuantizedTensor* wq = init_quantized_tensors((void**)&weight_q, 1, dim*dim);

    struct timeval tvs, tve;
    gettimeofday(&tvs, NULL);

    // float matmul
    for(int i=0; i<1000; i++) {
#ifdef NEON
        gemv_neon_float(result, x, weight, dim, dim);
#else
        matmul_float(result, x, weight, dim, dim);
#endif
    }
    gettimeofday(&tve, NULL);
    printf("float %ld ms\n", (tve.tv_sec*1000+tve.tv_usec/1000)-(tvs.tv_sec*1000+tvs.tv_usec/1000));

    float diff_max = 0;
    float diff_min = 0;
    for(int i=0; i<dim; i++) {
        diff_max = fmax(diff_max, result[i]-golden[i]);
        diff_min = fmin(diff_min, result[i]-golden[i]);
    }
    printf("%.5f %.5f\n", diff_max, diff_min);


    // W8A8 matmul
    gettimeofday(&tvs, NULL);

    QuantizedTensor xq = (QuantizedTensor) { .q = (int8_t*)calloc(dim, sizeof(int8_t)), .s = (float*)calloc(dim, sizeof(float)) };
    for(int i=0; i<1; i++) {
        quantize(&xq, x, dim);
#ifdef NEON
        matmul(result, &xq, wq, dim, dim);
        gemv_neon(result, &xq, wq, dim, dim);
#else
        matmul(result, &xq, wq, dim, dim);
#endif
    }
    gettimeofday(&tve, NULL);
    printf("W8A8 %ld ms\n", (tve.tv_sec*1000+tve.tv_usec/1000)-(tvs.tv_sec*1000+tvs.tv_usec/1000));


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