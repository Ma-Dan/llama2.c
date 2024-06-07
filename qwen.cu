/*
Inference for Llama-2 Transformer model in pure Cuda.
*/

#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <cub/cub.cuh>
#include <cublas_v2.h>

cublasHandle_t handle;

void createCublas() {
    cublasCreate(&handle);
}

void destroyCublas() {
    cublasDestroy(handle);
}

// ----------------------------------------------------------------------------
// GPU kernels

__global__ void element_wise_add_kernel(half* dest, half* src, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
        dest[i] = (half)((float)dest[i] + (float)src[i]);
}

__global__ void convert_fp32_to_fp16(half* out, float* in, int elements) {
    int index = blockIdx.x * 256 + threadIdx.x;
    if (index < elements)
        out[index] = (half)in[index];
}

__global__ void convert_fp16_to_fp32(float* out, half* in, int elements) {
    int index = blockIdx.x * 256 + threadIdx.x;
    if (index < elements)
        out[index] = (float)in[index];
}

// Single block - not enough parallelism for the GPU, but it's just 1% of total time
__global__ void rmsnorm_kernel(half* o, half* x, half* weight, int size, int elementsPerThread) {
    float ss = 0.0f;
    for (int i = 0; i < elementsPerThread; i++) {
        int index = threadIdx.x + i * 1024;
        if (index < size)
            ss += (float)x[index];
    }

    using BlockReduce = cub::BlockReduce<float, 1024>;
    __shared__ typename BlockReduce::TempStorage temp;
    ss = BlockReduce(temp).Sum(ss * ss);

    __shared__ float shared_ss;
    if (threadIdx.x == 0) {
        ss /= size;
        ss += 1e-6f;
        ss = 1.0f / sqrtf(ss);
        shared_ss = ss;
    }
    __syncthreads();
    ss = shared_ss;

    // normalize
    for (int i = 0; i < elementsPerThread; i++) {
        int index = threadIdx.x + i * 1024;
        if (index < size) {
            float val = (float)x[index];
            val *= ss * (float)weight[index];
            o[index] = (half)val;
        }
    }
}

// one output per warp so that we can parallelize the dot product across the warp
// Note that ~95% of total time is spent here, so optimizing this is important
__global__ void mat_vec_kernel(half* output, half* input, half* weight, int n, int d, int numSerialElements) {
    int index = blockIdx.x * blockDim.y + threadIdx.y;
    if (index >= d)
        return;

    float sum = 0;
    for (int i = 0; i < numSerialElements; i++) {
        int j = i * 32 + threadIdx.x;
        if (j < n)
            sum += ((float)weight[index * n + j]) * ((float)input[j]);
    }

    using WarpReduce = cub::WarpReduce<float>;
    __shared__ typename WarpReduce::TempStorage temp;
    sum = WarpReduce(temp).Sum(sum);

    if (threadIdx.x == 0)
        output[index] = (half)sum;
}

__global__ void mat_bias_vec_kernel(half* output, half* input, half* weight, half* bias, int n, int d, int numSerialElements) {
    int index = blockIdx.x * blockDim.y + threadIdx.y;
    if (index >= d)
        return;

    float sum = 0;
    for (int i = 0; i < numSerialElements; i++) {
        int j = i * 32 + threadIdx.x;
        if (j < n)
            sum += ((float)weight[index * n + j]) * ((float)input[j]);
    }

    using WarpReduce = cub::WarpReduce<float>;
    __shared__ typename WarpReduce::TempStorage temp;
    sum = WarpReduce(temp).Sum(sum);

    if (threadIdx.x == 0)
        output[index] = (half)(sum + (float)bias[index]);
}

// Each block processes a single head
__global__ void RoPERotation_kernel(half* sqk, half* f_real, half* f_imag, int num_heads, int head_size) {
    int h = blockIdx.x;
    half* qk = sqk + h * head_size;

    int i = threadIdx.x;
    float qk0 = qk[i];
    float qk1 = qk[i + head_size/2];
    float fcr = f_real[i];
    float fci = f_imag[i];
    qk[i] = qk0 * fcr - qk1 * fci;
    qk[i + head_size/2] = qk0 * fci + qk1 * fcr;
}

__device__ void softmax_gpu(float* __restrict__ x, int size) {
    using BlockReduce = cub::BlockReduce<float, 1024>;
    __shared__ typename BlockReduce::TempStorage temp;
    __shared__ float shared_val;

    int tid = threadIdx.x;
    int step = blockDim.x;

    // find max value (for numerical stability)
    float max_val = tid < size ? x[tid] : 0;
    for (int i = tid + step; i < size; i += step)
        if (x[i] > max_val)
            max_val = x[i];

    max_val = BlockReduce(temp).Reduce(max_val, cub::Max());
    if (threadIdx.x == 0)
        shared_val = max_val;
    __syncthreads();
    max_val = shared_val;

    // exp and sum
    float sum = 0.0f;
    for (int i = tid; i < size; i += step) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }

    sum = BlockReduce(temp).Sum(sum);
    if (threadIdx.x == 0)
        shared_val = sum;
    __syncthreads();
    sum = shared_val;

    // normalize
    for (int i = tid; i < size; i += step)
        x[i] /= sum;
}

// Each block processes a single head
// Poor parallelism and even poorer memory access pattern.
// Ankan - TODO: optimize this.
#define MAX_SEQ_LEN 8192
__global__ void MultiHeadAttention_kernel(half* __restrict__ output, const half* __restrict__ sq,
    const half* __restrict__ key_cache, const half* __restrict__ value_cache,
    int num_heads, int num_groups, int head_size, int kv_size, int loff, int seq_len, int dim) {
    int h = blockIdx.x;

    // get the query vector for this head
    const half* q = sq + h * head_size;
    // attention scores for this head
    __shared__ float att[MAX_SEQ_LEN];

    // iterate over all timesteps, including the current one
    for (int t = threadIdx.x; t < seq_len; t += blockDim.x) {
        // get the key vector for this head and at this timestep
        const half* k = key_cache + loff + t * kv_size + (h / num_groups) * head_size;
        // calculate the attention score as the dot product of q and k
        float score = 0.0f;
        for (int i = 0; i < head_size; i++)
            score += (float)q[i] * (float)k[i];
        score /= sqrtf(head_size);
        // save the score to the attention buffer
        att[t] = score;
    }
    __syncthreads();

    // softmax the scores to get attention weights
    softmax_gpu(att, seq_len);
    __syncthreads();

    // weighted sum of the values, store back into xb
    for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
        float val = 0.0f;
        for (int t = 0; t < seq_len; t++)
            val += att[t] * (float)value_cache[loff + t * kv_size + (h / num_groups) * head_size + i];
        output[h * head_size + i] = (half)val;
    }
}

__global__ void silu_element_wise_mul_kernel(half* dest, half* src, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        float val = (float)dest[i];
        val *= 1.0f / (1.0f + expf(-val));
        val *= (float)src[i];
        dest[i] = (half)val;
    }
}

// ----------------------------------------------------------------------------
// Transformer and RunState structs, and related memory management

typedef struct {
    int dim; // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int n_gqa_groups; // number of GQA share group
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len; // max sequence length
} Config;

typedef struct {
    // token embedding table
    half* token_embedding_table;    // (vocab_size, dim)
    // weights for rmsnorms
    half* rms_att_weight; // (layer, dim) rmsnorm weights
    half* rms_ffn_weight; // (layer, dim)
    // weights for matmuls
    half* wq; // (layer, dim, dim)
    half* wk; // (layer, dim, dim)
    half* wv; // (layer, dim, dim)
    half* wo; // (layer, dim, dim)
    // weights for ffn
    half* w1; // (layer, hidden_dim, dim)
    half* w2; // (layer, dim, hidden_dim)
    half* w3; // (layer, hidden_dim, dim)
    // final rmsnorm
    half* rms_final_weight; // (dim,)
    // freq_cis for RoPE relatively positional embeddings
    half* freq_cis_real; // (seq_len, dim/2)
    half* freq_cis_imag; // (seq_len, dim/2)
    // (optional) classifier weights for the logits, on the last layer
    half* wcls;
} TransformerWeights;

typedef struct {
    // current wave of activations
    half* x; // activation at current time stamp (dim,)
    half* xb; // same, but inside a residual branch (dim,)
    half* xb2; // an additional buffer just for convenience (dim,)
    half* hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    half* hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    half* q; // query (dim,)
    half* k; // key (dim,)
    half* v; // value (dim,)
    half* logits_gpu; // output logits
    float* logits_temp; // logits in GPU memory converted to float
    float* logits; // logits copied CPU side
    // kv cache
    half* key_cache;   // (layer, seq_len, dim)
    half* value_cache; // (layer, seq_len, dim)
} RunState;

void malloc_run_state(RunState* s, Config* p) {
    int kv_size = p->dim / p->n_gqa_groups;
    cudaMalloc((void**)&s->x, p->dim * sizeof(half));
    cudaMalloc((void**)&s->xb, p->dim * sizeof(half));
    cudaMalloc((void**)&s->xb2, p->dim * sizeof(half));
    cudaMalloc((void**)&s->hb, p->hidden_dim * sizeof(half));
    cudaMalloc((void**)&s->hb2, p->hidden_dim * sizeof(half));
    cudaMalloc((void**)&s->q, p->dim * sizeof(half));
    cudaMalloc((void**)&s->k, p->dim * sizeof(half));
    cudaMalloc((void**)&s->v, p->dim * sizeof(half));
    cudaMalloc((void**)&s->logits_gpu, p->vocab_size * sizeof(half));
    cudaMalloc((void**)&s->key_cache, p->n_layers * p->seq_len * kv_size * sizeof(half));    // potentially huge allocs
    cudaMalloc((void**)&s->value_cache, p->n_layers * p->seq_len * kv_size * sizeof(half));
    cudaMalloc((void**)&s->logits_temp, p->vocab_size * sizeof(float));
    s->logits = (float*)malloc(p->vocab_size * sizeof(float));

    // ensure all mallocs went fine
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q
        || !s->k || !s->v || !s->logits || !s->key_cache
        || !s->value_cache || !s->logits_gpu) {
        printf("malloc failed!\n");
        exit(1);
    }
}

void free_run_state(RunState* s) {
    cudaFree(s->x);
    cudaFree(s->xb);
    cudaFree(s->xb2);
    cudaFree(s->hb);
    cudaFree(s->hb2);
    cudaFree(s->q);
    cudaFree(s->k);
    cudaFree(s->v);
    cudaFree(s->logits_gpu);
    cudaFree(s->logits_temp);
    free(s->logits);
    cudaFree(s->key_cache);
    cudaFree(s->value_cache);
}

void malloc_weights(TransformerWeights* w, Config* p, int shared_weights) {
    int kv_size = p->dim / p->n_gqa_groups;
    cudaMalloc((void**)&w->token_embedding_table, p->vocab_size * p->dim * sizeof(half));
    cudaMalloc((void**)&w->rms_att_weight, p->n_layers * p->dim * sizeof(half));
    cudaMalloc((void**)&w->rms_ffn_weight, p->n_layers * p->dim * sizeof(half));
    cudaMalloc((void**)&w->wq, p->n_layers * (p->dim * p->dim + p->dim) * sizeof(half));
    cudaMalloc((void**)&w->wk, p->n_layers * (p->dim * kv_size + kv_size) * sizeof(half));
    cudaMalloc((void**)&w->wv, p->n_layers * (p->dim * kv_size + kv_size) * sizeof(half));
    cudaMalloc((void**)&w->wo, p->n_layers * p->dim * p->dim * sizeof(half));
    cudaMalloc((void**)&w->w1, p->n_layers * p->hidden_dim * p->dim * sizeof(half));
    cudaMalloc((void**)&w->w2, p->n_layers * p->dim * p->hidden_dim * sizeof(half));
    cudaMalloc((void**)&w->w3, p->n_layers * p->hidden_dim * p->dim * sizeof(half));
    cudaMalloc((void**)&w->rms_final_weight, p->dim * sizeof(half));
    int head_size = p->dim / p->n_heads;
    cudaMalloc((void**)&w->freq_cis_real, p->seq_len * head_size / 2 * sizeof(half));
    cudaMalloc((void**)&w->freq_cis_imag, p->seq_len * head_size / 2 * sizeof(half));

    if (shared_weights)
        w->wcls = w->token_embedding_table;
    else
        cudaMalloc((void**)&w->wcls, p->vocab_size * p->dim * sizeof(half));

    // ensure all mallocs went fine
    if (!w->token_embedding_table || !w->rms_att_weight || !w->rms_ffn_weight
        || !w->wq || !w->wk || !w->wv || !w->wo || !w->w1 || !w->w2 || !w->w3 ||
        !w->rms_final_weight || !w->freq_cis_real || !w->freq_cis_imag || !w->wcls) {
        printf("malloc failed!\n");
        exit(1);
    }
}

void free_weights(TransformerWeights* w, int shared_weights) {
    cudaFree(w->token_embedding_table);
    cudaFree(w->rms_att_weight);
    cudaFree(w->rms_ffn_weight);
    cudaFree(w->wq);
    cudaFree(w->wk);
    cudaFree(w->wv);
    cudaFree(w->wo);
    cudaFree(w->w1);
    cudaFree(w->w2);
    cudaFree(w->w3);
    cudaFree(w->rms_final_weight);
    cudaFree(w->freq_cis_real);
    cudaFree(w->freq_cis_imag);
    if (!shared_weights)
        cudaFree(w->wcls);
}

int divUp(int a, int b) {
    return (a - 1) / b + 1;
}

int uploadWeight(void *w, int elements, FILE* f, void *scratchCpu) {
    int count = fread(scratchCpu, sizeof(half), elements, f);
    if (count != elements) return 1;
    cudaMemcpyAsync(w, scratchCpu, sizeof(half) * elements, cudaMemcpyHostToDevice);
    return 0;
}

// ----------------------------------------------------------------------------
// initialization: read from checkpoint

int checkpoint_init_weights(TransformerWeights* w, Config* p, FILE* f, int shared_weights) {
    int kv_size = p->dim / p->n_gqa_groups;
    size_t scratch_size = p->n_layers * std::max(p->dim, p->hidden_dim) * p->dim;
    scratch_size = std::max((size_t)p->vocab_size * p->dim, scratch_size);
    scratch_size *= sizeof(half);
    void* scratchCpu = malloc(scratch_size);
    if (uploadWeight(w->token_embedding_table, p->vocab_size * p->dim, f, scratchCpu)) return 1;
    if (uploadWeight(w->rms_att_weight, p->n_layers * p->dim, f, scratchCpu)) return 1;
    if (uploadWeight(w->wq, p->n_layers * (p->dim * p->dim + p->dim), f, scratchCpu)) return 1;
    if (uploadWeight(w->wk, p->n_layers * (p->dim * kv_size + kv_size), f, scratchCpu)) return 1;
    if (uploadWeight(w->wv, p->n_layers * (p->dim * kv_size + kv_size), f, scratchCpu)) return 1;
    if (uploadWeight(w->wo, p->n_layers * p->dim * p->dim, f, scratchCpu)) return 1;
    if (uploadWeight(w->rms_ffn_weight, p->n_layers * p->dim, f, scratchCpu)) return 1;
    if (uploadWeight(w->w1, p->n_layers * p->dim * p->hidden_dim, f, scratchCpu)) return 1;
    if (uploadWeight(w->w2, p->n_layers * p->hidden_dim * p->dim, f, scratchCpu)) return 1;
    if (uploadWeight(w->w3, p->n_layers * p->dim * p->hidden_dim, f, scratchCpu)) return 1;
    if (uploadWeight(w->rms_final_weight, p->dim, f, scratchCpu)) return 1;

    int head_size = p->dim / p->n_heads;
    if (uploadWeight(w->freq_cis_real, p->seq_len * head_size / 2, f, scratchCpu)) return 1;
    if (uploadWeight(w->freq_cis_imag, p->seq_len * head_size / 2, f, scratchCpu)) return 1;

    if (!shared_weights)
        if (uploadWeight(w->wcls, p->vocab_size * p->dim, f, scratchCpu)) return 1;

    free(scratchCpu);
    return 0;
}

// ----------------------------------------------------------------------------
// neural net blocks

void accum(half* a, half* b, int size) {
    int blocks = divUp(size, 256);
    element_wise_add_kernel << <blocks, 256 >> > (a, b, size);
}


void rmsnorm(half* o, half* x, half* weight, int size) {
    int elementsPerThread = divUp(size, 1024);
    rmsnorm_kernel <<<1, 1024 >>> (o, x, weight, size, elementsPerThread);
}

void softmax(float* x, int size) {
    // find max value (for numerical stability)
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    // exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    // normalize
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

void matmul(half* xout, half* x, half* w, int n, int d) {
#if 1
    int serialElements = divUp(n, 32);
    dim3 block_dim(32, 4);
    int blocks = divUp(d, 4);
    mat_vec_kernel <<<blocks, block_dim >>> (xout, x, w, n, d, serialElements);
#else
    // Calculate with Cublas
    dim3 dimsA(n, d, 1);
    dim3 dimsB(1, n, 1);
    dim3 dimsC(dimsB.x, dimsA.y, 1);

    const half alpha = 1.0f;
    const half beta = 0.0f;
    cublasHgemm(
        handle, CUBLAS_OP_T, CUBLAS_OP_T, dimsA.y, dimsB.x,
        dimsA.x, &alpha, w, dimsA.x, x,
        dimsB.x, &beta, xout, dimsC.y);
#endif
}

void matmul_bias(half* xout, half* x, half* w, half* b, int n, int d) {
#if 1
    int serialElements = divUp(n, 32);
    dim3 block_dim(32, 4);
    int blocks = divUp(d, 4);
    mat_bias_vec_kernel <<<blocks, block_dim >>> (xout, x, w, b, n, d, serialElements);
#else
    // Calculate with Cublas
    dim3 dimsA(n, d, 1);
    dim3 dimsB(1, n, 1);
    dim3 dimsC(dimsB.x, dimsA.y, 1);

    const half alpha = 1.0f;
    const half beta = 1.0f;
    cudaMemcpyAsync(xout, b, d*sizeof(half), cudaMemcpyDeviceToDevice);
    cublasHgemm(
        handle, CUBLAS_OP_T, CUBLAS_OP_T, dimsA.y, dimsB.x,
        dimsA.x, &alpha, w, dimsA.x, x,
        dimsB.x, &beta, xout, dimsC.y);
#endif
}

void RoPERotation(half *qk, half *f_real, half *f_imag, int num_heads, int head_size) {
    RoPERotation_kernel <<<num_heads, head_size / 2 >>> (qk, f_real, f_imag, num_heads, head_size);
}

void MultiHeadAttention(half *output, half *q, half *key_cache, half *value_cache, int num_heads, int num_groups, int head_size, int kv_size, int loff, int seq_len) {
    int dim = head_size * num_heads;
    MultiHeadAttention_kernel <<<num_heads, 1024>>> (output, q, key_cache, value_cache, num_heads, num_groups, head_size, kv_size, loff, seq_len, dim);
}

void siluElementwiseMul(half *hb, half *hb2, int size) {
   silu_element_wise_mul_kernel <<<divUp(size, 256), 256 >>> (hb, hb2, size);
}

void transformer(int token, int pos, Config* p, RunState* s, TransformerWeights* w) {

    // a few convenience variables
    half* x = s->x;
    int dim = p->dim;
    int hidden_dim = p->hidden_dim;
    int head_size = dim / p->n_heads;
    int kv_size = dim / p->n_gqa_groups;

    // copy the token embedding into x
    half* content_row = &(w->token_embedding_table[token * dim]);
    cudaMemcpyAsync(x, content_row, dim * sizeof(half), cudaMemcpyDeviceToDevice);

    // pluck out the "pos" row of freq_cis_real and freq_cis_imag
    half* freq_cis_real_row = w->freq_cis_real + pos * head_size / 2;
    half* freq_cis_imag_row = w->freq_cis_imag + pos * head_size / 2;

    // forward all the layers
    for (int l = 0; l < p->n_layers; l++) {

        // attention rmsnorm
        rmsnorm(s->xb, x, w->rms_att_weight + l * dim, dim);

        // qkv matmuls for this position
        matmul_bias(s->q, s->xb, w->wq + l * (dim * dim + dim), w->wq + l * (dim * dim + dim) + dim * dim, dim, dim);
        matmul_bias(s->k, s->xb, w->wk + l * (dim * kv_size + kv_size), w->wk + l * (dim * kv_size + kv_size) + dim * kv_size, dim, kv_size);
        matmul_bias(s->v, s->xb, w->wv + l * (dim * kv_size + kv_size), w->wv + l * (dim * kv_size + kv_size) + dim * kv_size, dim, kv_size);

        // apply RoPE rotation to the q and k vectors for each head
        RoPERotation(s->q, freq_cis_real_row, freq_cis_imag_row, p->n_heads, head_size);
        RoPERotation(s->k, freq_cis_real_row, freq_cis_imag_row, kv_size/head_size, head_size);

        // save key,value at this time step (pos) to our kv cache
        int loff = l * p->seq_len * kv_size; // kv cache layer offset for convenience
        half* key_cache_row = s->key_cache + loff + pos * kv_size;
        half* value_cache_row = s->value_cache + loff + pos * kv_size;
        cudaMemcpyAsync(key_cache_row, s->k, kv_size * sizeof(half), cudaMemcpyDeviceToDevice);
        cudaMemcpyAsync(value_cache_row, s->v, kv_size * sizeof(half), cudaMemcpyDeviceToDevice);

        MultiHeadAttention(s->xb, s->q, s->key_cache, s->value_cache, p->n_heads, p->n_gqa_groups, head_size, kv_size, loff, pos+1);

        // final matmul to get the output of the attention
        matmul(s->xb2, s->xb, w->wo + l * dim * dim, dim, dim);

        // residual connection back into x
        accum(x, s->xb2, dim);

        // ffn rmsnorm
        rmsnorm(s->xb, x, w->rms_ffn_weight + l * dim, dim);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        matmul(s->hb, s->xb, w->w1 + l * dim * hidden_dim, dim, hidden_dim);
        matmul(s->hb2, s->xb, w->w3 + l * dim * hidden_dim, dim, hidden_dim);

        // apply F.silu activation on hb and multiply it with hb2
        siluElementwiseMul(s->hb, s->hb2, hidden_dim);

        // final matmul to get the output of the ffn
        matmul(s->xb, s->hb, w->w2 + l * dim * hidden_dim, hidden_dim, dim);

        // residual connection
        accum(x, s->xb, dim);
    }

    // final rmsnorm
    rmsnorm(x, x, w->rms_final_weight, dim);

    // classifier into logits
    matmul(s->logits_gpu, x, w->wcls, p->dim, p->vocab_size);

    // copy logits from GPU->CPU
    convert_fp16_to_fp32 <<<divUp(p->vocab_size, 256), 256 >>> (s->logits_temp, s->logits_gpu, p->vocab_size);
    cudaMemcpy(s->logits, s->logits_temp, p->vocab_size * sizeof(float), cudaMemcpyDeviceToHost);
}

// ----------------------------------------------------------------------------
// byte pair encoding (BPE) tokenizer, encodes strings into tokens so we can prompt

typedef struct {
    char *str;
    int id;
} TokenIndex;

int compare_tokens(const void *a, const void *b) {
    return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}

int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
    // efficiently find the perfect match for str in vocab, return its index or -1 if not found
    TokenIndex tok = { .str = str }; // acts as the key to search for
    TokenIndex *res = (TokenIndex *)bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}

void bpe_encode(char *text, char **vocab, float *vocab_scores, int vocab_size, unsigned int max_token_length, int *tokens, int *n_tokens) {

    // sort vocabulary
    TokenIndex *sorted_vocab = (TokenIndex *)malloc(vocab_size * sizeof(TokenIndex));
    for (int i = 0; i < vocab_size; i++) {
        sorted_vocab[i].str = vocab[i];
        sorted_vocab[i].id = i;
    }
    qsort(sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);

    // create a temporary buffer that will store merge candidates of always two consecutive tokens
    char* str_buffer = (char *)malloc((max_token_length*2 +1 +2) * sizeof(char)); // *2 for concat, +1 for null terminator +2 for UTF8 (in case max_token_lenght is 1)
    size_t str_len = 0;

    // add_dummy_prefix is true by default
    tokens[0] = str_lookup(" ", sorted_vocab, vocab_size);
    *n_tokens = 1; // the number of tokens

    // Okay UTF-8 time. This will get messy. Here is the reference from Wikipedia:
    // Code point ↔ UTF-8 conversion
    // First code point	Last code point	Byte 1	Byte 2	Byte 3	Byte 4
    // U+0000	U+007F	    0xxxxxxx
    // U+0080	U+07FF	    110xxxxx	10xxxxxx
    // U+0800	U+FFFF	    1110xxxx	10xxxxxx	10xxxxxx
    // U+10000	U+10FFFF    11110xxx	10xxxxxx	10xxxxxx	10xxxxxx

    // process the raw (UTF-8) byte sequence of the input string
    for (char *c = text; *c != '\0'; c++) {

        // reset buffer if the current byte is ASCII or a leading byte
        // 0xC0 is 11000000, so (*c & 0xC0) keeps the first 2 bits and zeros the rest
        // 0x80 is 10000000
        // in UTF-8, all continuation bytes start with "10" in first two bits
        // so in English this is: "if this byte is not a continuation byte"
        if ((*c & 0xC0) != 0x80) {
            // this byte must be either a leading byte (11...) or an ASCII char (0x...)
            // => reset our location, as we're starting a new UTF-8 codepoint
            str_len = 0;
        }

        // append the current byte to the buffer
        str_buffer[str_len++] = *c; // ++ is post-increment, incremented after this line
        str_buffer[str_len] = '\0';

        // while the next character is a continuation byte, continue appending
        // but if there are too many of them, just stop to avoid overruning str_buffer size.
        if ((*(c+1) & 0xC0) == 0x80 && str_len < 4) {
            continue;
        }

        // ok c+1 is not a continuation byte, so we've read in a full codepoint
        int id = str_lookup(str_buffer, sorted_vocab, vocab_size);

        if (id != -1) {
            // we found this codepoint in vocab, add it as a token
            tokens[(*n_tokens)++] = id;
        } else {
            // byte_fallback encoding: just encode each byte as a token
            // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
            // so the individual bytes only start at index 3
            for (int i=0; i < str_len; i++) {
                tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
            }
        }
        str_len = 0; // protect against a sequence of stray UTF8 continuation bytes
    }

    // merge the best consecutive pair each iteration, according the scores in vocab_scores
    while (1) {
        float best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;

        for (int i=0; i < (*n_tokens-1); i++) {
            // check if we can merge the pair (tokens[i], tokens[i+1])
            sprintf(str_buffer, "%s%s", vocab[tokens[i]], vocab[tokens[i+1]]);
            int id = str_lookup(str_buffer, sorted_vocab, vocab_size);
            if (id != -1 && vocab_scores[id] > best_score) {
                // this merge pair exists in vocab! record its score and position
                best_score = vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1) {
            break; // we couldn't find any more pairs to merge, so we're done
        }

        // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
        tokens[best_idx] = best_id;
        // delete token at position best_idx+1, shift the entire sequence back 1
        for (int i = best_idx+1; i < (*n_tokens-1); i++) {
            tokens[i] = tokens[i+1];
        }
        (*n_tokens)--; // token length decreased
    }

    free(str_buffer);
    free(sorted_vocab);
}

// ----------------------------------------------------------------------------
// utilities

long time_in_ms() {
    // return time in milliseconds, for benchmarking the model speed
    struct timespec time;
    timespec_get(&time, TIME_UTC);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

unsigned long long rng_seed;
unsigned int random_u32() {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    rng_seed ^= rng_seed >> 12;
    rng_seed ^= rng_seed << 25;
    rng_seed ^= rng_seed >> 27;
    return (rng_seed * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32() { // random float32 in [0,1)
    return (random_u32() >> 8) / 16777216.0f;
}

int sample(float* probabilities, int n) {
    // sample index from probabilities, they must sum to 1
    float r = random_f32();
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (r < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

int argmax(float* v, int n) {
    // return argmax of v in elements 0..n
    int max_i = 0;
    float max_p = v[0];
    for (int i = 1; i < n; i++) {
        if (v[i] > max_p) {
            max_i = i;
            max_p = v[i];
        }
    }
    return max_i;
}
// ----------------------------------------------------------------------------

int main(int argc, char *argv[]) {

    // poor man's C argparse
    char *checkpoint = NULL;  // e.g. out/model.bin
    float temperature = 0.9f; // e.g. 1.0, or 0.0
    int steps = 256;          // max number of steps to run for, 0: use seq_len
    char *prompt = NULL;      // prompt string

    // 'checkpoint' is necessary arg
    if (argc < 2) {
        printf("Usage: %s <checkpoint_file> [temperature] [steps] [prompt]\n", argv[0]);
        return 1;
    }
    if (argc >= 2) {
        checkpoint = argv[1];
    }
    if (argc >= 3) {
        // optional temperature. 0.0 = (deterministic) argmax sampling. 1.0 = baseline
        temperature = atof(argv[2]);
    }
    if (argc >= 4) {
        steps = atoi(argv[3]);
    }
    if (argc >= 5) {
        prompt = argv[4];
    }

    // seed rng with time. if you want deterministic behavior use temperature 0.0
    rng_seed = (unsigned int)time(NULL);

    createCublas();

    // read in the model.bin file
    Config config;
    TransformerWeights weights;
    int shared_weights;
    {
        FILE *file = fopen(checkpoint, "rb");
        if (!file) { printf("Couldn't open file %s\n", checkpoint); return 1; }
        // read in the config header
        if (fread(&config, sizeof(Config), 1, file) != 1) { return 1; }

        // Dump model config
        printf("\nModel params:- \ndim: %d \nhidden_dim: %d\nn_heads: %d\nn_kv_heads: %d\nn_layers: %d\nseq_len: %d\nvocab_size: %d\n\n",
            config.dim, config.hidden_dim, config.n_heads, config.n_kv_heads, config.n_layers, config.seq_len, config.vocab_size);

        // negative vocab size is hacky way of signaling unshared weights. bit yikes.
        shared_weights = config.vocab_size > 0 ? 1 : 0;
        config.vocab_size = abs(config.vocab_size);
        // read in the Transformer weights
        malloc_weights(&weights, &config, shared_weights);
        if (checkpoint_init_weights(&weights, &config, file, shared_weights)) { return 1; }
    }
    // right now we cannot run for more than config.seq_len steps
    if (steps <= 0 || steps > config.seq_len) { steps = config.seq_len; }

    // read in the tokenizer.bin file
    char** vocab = (char**)malloc(config.vocab_size * sizeof(char*));
    float* vocab_scores = (float*)malloc(config.vocab_size * sizeof(float));
    unsigned int max_token_length;
    {
        FILE *file = fopen("qwen_chat_tokenizer.bin", "rb");
        if (!file) { printf("couldn't load tokenizer.bin\n"); return 1; }
        if (fread(&max_token_length, sizeof(int), 1, file) != 1) { printf("failed read\n"); return 1; }
        int len;
        for (int i = 0; i < config.vocab_size; i++) {
            if (fread(vocab_scores + i, sizeof(float), 1, file) != 1) { printf("failed read\n"); return 1;}
            if (fread(&len, sizeof(int), 1, file) != 1) { printf("failed read\n"); return 1; }
            vocab[i] = (char *)malloc(len + 1);
            if (len>0 && (fread(vocab[i], len, 1, file) != 1)) { printf("failed read\n"); return 1; }
            vocab[i][len] = '\0'; // add the string terminating token
        }
        fclose(file);
    }

    // create and init the application RunState
    RunState state;
    malloc_run_state(&state, &config);

    // process the prompt, if any
    /*int *prompt_tokens = NULL;
    int num_prompt_tokens = 0;
    if (prompt != NULL) {
        prompt_tokens = (int*)malloc(config.seq_len * sizeof(int));
        bpe_encode(prompt, vocab, vocab_scores, config.vocab_size, max_token_length, prompt_tokens, &num_prompt_tokens);
    }*/
    //int *prompt_tokens = (int*)malloc(21*sizeof(int));
    int num_prompt_tokens = 21;
    int prompt_tokens[21] = {151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198, 109432, 99283, 151645, 198, 151644, 77091, 198};

    // start the main loop
    long start = 0;  // used to time our code, only initialized after first iteration
    int next;        // will store the next token in the sequence
    int token = prompt_tokens[0];
    int pos = 1;     // position in the sequence
    while (pos < steps) {

        // forward the transformer to get logits for the next token
        transformer(token, pos, &config, &state, &weights);

        if(pos < num_prompt_tokens) {
            // if we are still processing the input prompt, force the next prompt token
            next = prompt_tokens[pos];
        } else {
            // sample the next token
            if (temperature == 0.0f) {
                // greedy argmax sampling: take the token with the highest probability
                next = argmax(state.logits, config.vocab_size);
            } else {
                // apply the temperature to the logits
                for (int q=0; q<config.vocab_size; q++) { state.logits[q] /= temperature; }
                // apply softmax to the logits to get the probabilities for next token
                softmax(state.logits, config.vocab_size);
                // we sample from this distribution to get the next token
                next = sample(state.logits, config.vocab_size);
            }
        }

        // following BOS token (1), sentencepiece decoder strips any leading whitespace (see PR #89)
        char *token_str = (token == 1 && vocab[next][0] == ' ') ? vocab[next]+1 : vocab[next];
        printf("%s", token_str);
        fflush(stdout);

        if (next == 2) break; // break if EOS token is reached

        // advance forward
        token = next;
        pos++;
        // init our timer here because the first iteration could be slow
        if (start == 0) { start = time_in_ms(); }
    }

    // report achieved tok/s
    long end = time_in_ms();
    double time = (end - start) / 1000.0;
    int timed_tokens = pos - 1;
    printf("\nachieved tok/s: %f. Tokens: %d, seconds: %g\n", timed_tokens / time, timed_tokens, time);

    destroyCublas();

    // memory cleanup
    free_run_state(&state);
    free_weights(&weights, shared_weights);
    for (int i = 0; i < config.vocab_size; i++) { free(vocab[i]); }
    free(vocab);
    free(vocab_scores);
    //if (prompt_tokens != NULL) free(prompt_tokens);
    return 0;
}
