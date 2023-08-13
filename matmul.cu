/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * Matrix multiplication: C = A * B.
 * Host code.
 *
 * This sample implements matrix multiplication which makes use of shared memory
 * to ensure data reuse, the matrix multiplication is done using tiling approach.
 * It has been written for clarity of exposition to illustrate various CUDA programming
 * principles, not with the goal of providing the most performant generic kernel for matrix multiplication.
 * See also:
 * V. Volkov and J. Demmel, "Benchmarking GPUs to tune dense linear algebra,"
 * in Proc. 2008 ACM/IEEE Conf. on Supercomputing (SC '08),
 * Piscataway, NJ: IEEE Press, 2008, pp. Art. 31:1-11.
 */

// System includes
#include <stdio.h>
#include <stdint.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <cublas_v2.h>

// Beginning of GPU Architecture definitions
inline int _ConvertSMVer2Cores(int major, int minor) {
  // Defines for GPU Architecture types (using the SM version to determine
  // the # of cores per SM
  typedef struct {
    int SM;  // 0xMm (hexidecimal notation), M = SM Major version,
    // and m = SM minor version
    int Cores;
  } sSMtoCores;

  sSMtoCores nGpuArchCoresPerSM[] = {
      {0x30, 192},
      {0x32, 192},
      {0x35, 192},
      {0x37, 192},
      {0x50, 128},
      {0x52, 128},
      {0x53, 128},
      {0x60,  64},
      {0x61, 128},
      {0x62, 128},
      {0x70,  64},
      {0x72,  64},
      {0x75,  64},
      {0x80,  64},
      {0x86, 128},
      {0x87, 128},
      {0x89, 128},
      {0x90, 128},
      {-1, -1}};

  int index = 0;

  while (nGpuArchCoresPerSM[index].SM != -1) {
    if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
      return nGpuArchCoresPerSM[index].Cores;
    }

    index++;
  }

  // If we don't find the values, we default use the previous one
  // to run properly
  printf(
      "MapSMtoCores for SM %d.%d is undefined."
      "  Default to use %d Cores/SM\n",
      major, minor, nGpuArchCoresPerSM[index - 1].Cores);
  return nGpuArchCoresPerSM[index - 1].Cores;
}

static const char *_cudaGetErrorEnum(cudaError_t error) {
  return cudaGetErrorName(error);
}

inline const char* _ConvertSMVer2ArchName(int major, int minor) {
  // Defines for GPU Architecture types (using the SM version to determine
  // the GPU Arch name)
  typedef struct {
    int SM;  // 0xMm (hexidecimal notation), M = SM Major version,
    // and m = SM minor version
    const char* name;
  } sSMtoArchName;

  sSMtoArchName nGpuArchNameSM[] = {
      {0x30, "Kepler"},
      {0x32, "Kepler"},
      {0x35, "Kepler"},
      {0x37, "Kepler"},
      {0x50, "Maxwell"},
      {0x52, "Maxwell"},
      {0x53, "Maxwell"},
      {0x60, "Pascal"},
      {0x61, "Pascal"},
      {0x62, "Pascal"},
      {0x70, "Volta"},
      {0x72, "Xavier"},
      {0x75, "Turing"},
      {0x80, "Ampere"},
      {0x86, "Ampere"},
      {0x87, "Ampere"},
      {0x89, "Ada"},
      {0x90, "Hopper"},
      {-1, "Graphics Device"}};

  int index = 0;

  while (nGpuArchNameSM[index].SM != -1) {
    if (nGpuArchNameSM[index].SM == ((major << 4) + minor)) {
      return nGpuArchNameSM[index].name;
    }

    index++;
  }

  // If we don't find the values, we default use the previous one
  // to run properly
  printf(
      "MapSMtoArchName for SM %d.%d is undefined."
      "  Default to use %s\n",
      major, minor, nGpuArchNameSM[index - 1].name);
  return nGpuArchNameSM[index - 1].name;
}

template <typename T>
void check(T result, char const *const func, const char *const file,
           int const line) {
  if (result) {
    fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
    exit(EXIT_FAILURE);
  }
}

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

/**
 * Matrix multiplication (CUDA Kernel) on the device: C = A * B
 * wA is A's width and wB is B's width
 */
template <int BLOCK_SIZE> __global__ void MatrixMulCUDA(float *C, float *A,
    float *B, int wA,
    int wB) {
  // Block index
  int bx = blockIdx.x;
  int by = blockIdx.y;

  // Thread index
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // Index of the first sub-matrix of A processed by the block
  int aBegin = wA * BLOCK_SIZE * by;

  // Index of the last sub-matrix of A processed by the block
  int aEnd   = aBegin + wA - 1;

  // Step size used to iterate through the sub-matrices of A
  int aStep  = BLOCK_SIZE;

  // Index of the first sub-matrix of B processed by the block
  int bBegin = BLOCK_SIZE * bx;

  // Step size used to iterate through the sub-matrices of B
  int bStep  = BLOCK_SIZE * wB;

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  float Csub = 0;

  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (int a = aBegin, b = bBegin;
       a <= aEnd;
       a += aStep, b += bStep) {
    // Declaration of the shared memory array As used to
    // store the sub-matrix of A
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

    // Declaration of the shared memory array Bs used to
    // store the sub-matrix of B
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    // Load the matrices from device memory
    // to shared memory; each thread loads
    // one element of each matrix
    As[ty][tx] = A[a + wA * ty + tx];
    Bs[ty][tx] = B[b + wB * ty + tx];

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

    // Multiply the two matrices together;
    // each thread computes one element
    // of the block sub-matrix
#pragma unroll

    for (int k = 0; k < BLOCK_SIZE; ++k) {
      Csub += As[ty][k] * Bs[k][tx];
    }

    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    __syncthreads();
  }

  // Write the block sub-matrix to device memory;
  // each thread writes one element
  int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
  C[c + wB * ty + tx] = Csub;
}


// This function returns the best GPU (with maximum GFLOPS)
inline int gpuGetMaxGflopsDeviceId() {
  int current_device = 0, sm_per_multiproc = 0;
  int max_perf_device = 0;
  int device_count = 0;
  int devices_prohibited = 0;

  uint64_t max_compute_perf = 0;
  checkCudaErrors(cudaGetDeviceCount(&device_count));

  if (device_count == 0) {
    fprintf(stderr,
            "gpuGetMaxGflopsDeviceId() CUDA error:"
            " no devices supporting CUDA.\n");
    exit(EXIT_FAILURE);
  }

  // Find the best CUDA capable GPU device
  current_device = 0;

  while (current_device < device_count) {
    int computeMode = -1, major = 0, minor = 0;
    checkCudaErrors(cudaDeviceGetAttribute(&computeMode, cudaDevAttrComputeMode, current_device));
    checkCudaErrors(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, current_device));
    checkCudaErrors(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, current_device));

    // If this GPU is not running on Compute Mode prohibited,
    // then we can add it to the list
    if (computeMode != cudaComputeModeProhibited) {
      if (major == 9999 && minor == 9999) {
        sm_per_multiproc = 1;
      } else {
        sm_per_multiproc =
            _ConvertSMVer2Cores(major,  minor);
      }
      int multiProcessorCount = 0, clockRate = 0;
      checkCudaErrors(cudaDeviceGetAttribute(&multiProcessorCount, cudaDevAttrMultiProcessorCount, current_device));
      cudaError_t result = cudaDeviceGetAttribute(&clockRate, cudaDevAttrClockRate, current_device);
      if (result != cudaSuccess) {
        // If cudaDevAttrClockRate attribute is not supported we
        // set clockRate as 1, to consider GPU with most SMs and CUDA Cores.
        if(result == cudaErrorInvalidValue) {
          clockRate = 1;
        }
        else {
          fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \n", __FILE__, __LINE__,
            static_cast<unsigned int>(result), _cudaGetErrorEnum(result));
          exit(EXIT_FAILURE);
        }
      }
      uint64_t compute_perf = (uint64_t)multiProcessorCount * sm_per_multiproc * clockRate;

      if (compute_perf > max_compute_perf) {
        max_compute_perf = compute_perf;
        max_perf_device = current_device;
      }
    } else {
      devices_prohibited++;
    }

    ++current_device;
  }

  if (devices_prohibited == device_count) {
    fprintf(stderr,
            "gpuGetMaxGflopsDeviceId() CUDA error:"
            " all devices have compute mode prohibited.\n");
    exit(EXIT_FAILURE);
  }

  return max_perf_device;
}

// Initialization code to find the best CUDA Device
inline int findCudaDevice(int argc, const char **argv) {
  int devID = 0;

  // Otherwise pick the device with highest Gflops/s
  devID = gpuGetMaxGflopsDeviceId();
  checkCudaErrors(cudaSetDevice(devID));
  int major = 0, minor = 0;
  checkCudaErrors(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, devID));
  checkCudaErrors(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, devID));
  printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n",
         devID, _ConvertSMVer2ArchName(major, minor), major, minor);

  return devID;
}


// Global resource
cudaStream_t stream;
cublasHandle_t handle;

void createResource() {
    checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cublasCreate(&handle);
    cublasSetStream(handle, stream);
}

void destroyResource() {
    checkCudaErrors(cudaStreamDestroy(stream));
    cublasDestroy(handle);
}


// Hard code memory
float *h_A_288_288;
float *h_A_288_768;
float *h_A_288_32000;
float *h_B_288;
float *h_B_768;
float *h_C_288;
float *h_C_768;
float *h_C_32000;

float *d_A_288_288;
float *d_A_288_768;
float *d_A_288_32000;
float *d_B_288;
float *d_B_768;
float *d_C_288;
float *d_C_768;
float *d_C_32000;

void allocateMemory() {
    checkCudaErrors(cudaMallocHost(&h_A_288_288, 288*288*sizeof(float)));
    checkCudaErrors(cudaMallocHost(&h_A_288_768, 288*768*sizeof(float)));
    checkCudaErrors(cudaMallocHost(&h_A_288_32000, 288*32000*sizeof(float)));

    checkCudaErrors(cudaMallocHost(&h_B_288, 288*sizeof(float)));
    checkCudaErrors(cudaMallocHost(&h_B_768, 768*sizeof(float)));

    checkCudaErrors(cudaMallocHost(&h_C_288, 288*sizeof(float)));
    checkCudaErrors(cudaMallocHost(&h_C_768, 768*sizeof(float)));
    checkCudaErrors(cudaMallocHost(&h_C_32000, 32000*sizeof(float)));

    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_A_288_288), 288*288*sizeof(float)));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_A_288_768), 288*768*sizeof(float)));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_A_288_32000), 288*32000*sizeof(float)));

    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_B_288), 288*sizeof(float)));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_B_768), 768*sizeof(float)));

    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_C_288), 288*sizeof(float)));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_C_768), 768*sizeof(float)));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_C_32000), 32000*sizeof(float)));
}

void freeMemory() {
    checkCudaErrors(cudaFreeHost(h_A_288_288));
    checkCudaErrors(cudaFreeHost(h_A_288_768));
    checkCudaErrors(cudaFreeHost(h_A_288_32000));

    checkCudaErrors(cudaFreeHost(h_B_288));
    checkCudaErrors(cudaFreeHost(h_B_768));

    checkCudaErrors(cudaFreeHost(h_C_288));
    checkCudaErrors(cudaFreeHost(h_C_768));
    checkCudaErrors(cudaFreeHost(h_C_32000));

    checkCudaErrors(cudaFree(d_A_288_288));
    checkCudaErrors(cudaFree(d_A_288_768));
    checkCudaErrors(cudaFree(d_A_288_32000));

    checkCudaErrors(cudaFree(d_B_288));
    checkCudaErrors(cudaFree(d_B_768));

    checkCudaErrors(cudaFree(d_C_288));
    checkCudaErrors(cudaFree(d_C_768));
    checkCudaErrors(cudaFree(d_C_32000));
}

// Cublas
void matmul(float* xout, float* x, float* w, int n, int d) {
    dim3 dimsA(n, d, 1);
    dim3 dimsB(1, n, 1);
    dim3 dimsC(dimsB.x, dimsA.y, 1);

    int mem_size_A = n*d*sizeof(float);
    int mem_size_B = n*sizeof(float);
    int mem_size_C = d*sizeof(float);

    float *h_A;
    float *d_A;
    if(n==288 && d==288) {
        h_A = h_A_288_288;
        d_A = d_A_288_288;
    }
    if(n==288 && d==768) {
        h_A = h_A_288_768;
        d_A = d_A_288_768;
    }
    if(n==768 && d==288) {
        h_A = h_A_288_768;
        d_A = d_A_288_768;
    }
    if(n==288 && d==32000) {
        h_A = h_A_288_32000;
        d_A = d_A_288_32000;
    }

    float *h_B;
    float *d_B;
    if(n==288) {
        h_B = h_B_288;
        d_B = d_B_288;
    }
    if(n==768) {
        h_B = h_B_768;
        d_B = d_B_768;
    }

    float *h_C;
    float *d_C;
    if(d==288) {
        h_C = h_C_288;
        d_C = d_C_288;
    }
    if(d==768) {
        h_C = h_C_768;
        d_C = d_C_768;
    }
    if(d==32000) {
        h_C = h_C_32000;
        d_C = d_C_32000;
    }

    memcpy(h_A, w, mem_size_A);
    memcpy(h_B, x, mem_size_B);

    // copy host memory to device
    checkCudaErrors(cudaMemcpyAsync(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice, stream));

    // Calculate with Cublas
    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasStatus_t status = cublasSgemm(
        handle, CUBLAS_OP_T, CUBLAS_OP_T, dimsA.y, dimsB.x,
        dimsA.x, &alpha, d_A, dimsA.x, d_B,
        dimsB.x, &beta, d_C, dimsC.y);
    /*cublasSgemm(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, dimsB.x, dimsA.y,
        dimsA.x, &alpha, d_B, dimsB.x, d_A,
        dimsA.x, &beta, d_C, dimsB.x);*/

    // Copy result from device to host
    checkCudaErrors(cudaMemcpyAsync(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost, stream));
    checkCudaErrors(cudaStreamSynchronize(stream));

    memcpy(xout, h_C, mem_size_C);
}


// 1D kernel
__global__ void MatMulKernel1D(float *C, float *A, float *B, const int wh, const int wC, const int hC)
{
    const int totalSize = wC * hC;
    int thID = threadIdx.x + blockIdx.x * blockDim.x;
    while (thID < totalSize) {
        int Cx = thID / wC;
        int Cy = thID % wC;
        float rst = 0.0;
        for (int i = 0; i < wh; i++) {
            rst += A[Cx * wh + i] * B[i * wC + Cy];
        }
        C[Cx * wC + Cy] = rst;
        thID += gridDim.x * blockDim.x;
    }
    __syncthreads();
}

void matmul_1(float* xout, float* x, float* w, int n, int d) {
    //printf("matmul %d, %d\n", n, d);
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    /*int blockSize = 32;
    int threadsPerBlock = blockSize * blockSize;*/

    int threadSize = 288;

    dim3 dimsA(n, d, 1);
    dim3 dimsB(1, n, 1);
    dim3 dimsC(dimsB.x, dimsA.y, 1);

    int mem_size_A = n*d*sizeof(float);
    int mem_size_B = n*sizeof(float);
    int mem_size_C = d*sizeof(float);

    float *h_A;
    float *d_A;
    if(n==288 && d==288) {
        h_A = h_A_288_288;
        d_A = d_A_288_288;
    }
    if(n==288 && d==768) {
        h_A = h_A_288_768;
        d_A = d_A_288_768;
    }
    if(n==768 && d==288) {
        h_A = h_A_288_768;
        d_A = d_A_288_768;
    }
    if(n==288 && d==32000) {
        h_A = h_A_288_32000;
        d_A = d_A_288_32000;
    }

    float *h_B;
    float *d_B;
    if(n==288) {
        h_B = h_B_288;
        d_B = d_B_288;
    }
    if(n==768) {
        h_B = h_B_768;
        d_B = d_B_768;
    }

    float *h_C;
    float *d_C;
    if(d==288) {
        h_C = h_C_288;
        d_C = d_C_288;
    }
    if(d==768) {
        h_C = h_C_768;
        d_C = d_C_768;
    }
    if(d==32000) {
        h_C = h_C_32000;
        d_C = d_C_32000;
    }

    memcpy(h_A, w, mem_size_A);
    memcpy(h_B, x, mem_size_B);

    // copy host memory to device
    checkCudaErrors(cudaMemcpyAsync(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice, stream));

    // Setup execution parameters
    int grid = dimsC.x * dimsC.y / threadSize;

    MatMulKernel1D<<<grid, threadSize, 0, stream>>>(d_C, d_A, d_B, dimsA.x, dimsC.x, dimsC.y);

    // Copy result from device to host
    checkCudaErrors(cudaMemcpyAsync(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost, stream));
    checkCudaErrors(cudaStreamSynchronize(stream));

    memcpy(xout, h_C, mem_size_C);
}


// 2D kernel
template <int BLOCK_SIZE> __global__ void MatMulKernel2DAnySize(float *C, float *A, float *B, int wA, int wC, int hC)
{
    int wB = wC;
    int maxIdxA = wA * hC;

    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;
    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    while (wA * BLOCK_SIZE * by < maxIdxA) {
        // Index of the first sub-matrix of A processed by the block
        int aBegin = wA * BLOCK_SIZE * by;

        // Index of the last sub-matrix of A processed by the block
        int aEnd = aBegin + wA - 1;

        // Step size used to iterate through the sub-matrices of A
        int aStep = BLOCK_SIZE;

        // Index of the first sub-matrix of B processed by the block
        int bBegin = BLOCK_SIZE * bx;

        // Step size used to iterate through the sub-matrices of B
        int bStep = BLOCK_SIZE * wB;

        // Csub is used to store the element of the block sub-matrix
        // that is computed by the thread
        float Csub = 0;

        // Loop over all the sub-matrices of A and B
        // required to compute the block sub-matrix
        int flag = 0;
        for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
            // Declaration of the shared memory array As used to
            // store the sub-matrix of A
            __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

            // Declaration of the shared memory array Bs used to
            // store the sub-matrix of B
            __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

            // Load the matrices from device memory
            // to shared memory; each thread loads
            // one element of each matrix
            if (flag * BLOCK_SIZE + tx < wA || flag * BLOCK_SIZE + ty < hC) {
                As[ty][tx] = A[a + wA * ty + tx];
            } else {
                As[ty][tx] = 0.0;
            }

            if (flag * BLOCK_SIZE + ty < wA || flag * BLOCK_SIZE + tx < wC) {
                Bs[ty][tx] = B[b + wB * ty + tx];
            } else {
                Bs[ty][tx] = 0.0;
            }

            // Synchronize to make sure the matrices are loaded
            __syncthreads();

            // Multiply the two matrices together;
            // each thread computes one element
            // of the block sub-matrix
#pragma unroll

            for (int k = 0; k < BLOCK_SIZE; ++k) {
                Csub += As[ty][k] * Bs[k][tx];
            }

            // Synchronize to make sure that the preceding
            // computation is done before loading two new
            // sub-matrices of A and B in the next iteration
            __syncthreads();
            flag++;
        }

        // Write the block sub-matrix to device memory;
        // each thread writes one element
        if (BLOCK_SIZE * bx + tx < wC && BLOCK_SIZE * by + ty < hC) { // thread could over max.
            C[wB * BLOCK_SIZE * by + BLOCK_SIZE * bx + wB * ty + tx] = Csub;
        }
        bx += BLOCK_SIZE;
        by += BLOCK_SIZE;
    }
}

void matmul_0(float* xout, float* x, float* w, int n, int d) {
    //printf("matmul %d, %d\n", n, d);
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    dim3 dimsA(n, d, 1);
    dim3 dimsB(1, n, 1);
    dim3 dimsC(dimsB.x, dimsA.y, 1);

    int mem_size_A = n*d*sizeof(float);
    int mem_size_B = n*sizeof(float);
    int mem_size_C = d*sizeof(float);

    float *h_A;
    float *d_A;
    if(n==288 && d==288) {
        h_A = h_A_288_288;
        d_A = d_A_288_288;
    }
    if(n==288 && d==768) {
        h_A = h_A_288_768;
        d_A = d_A_288_768;
    }
    if(n==768 && d==288) {
        h_A = h_A_288_768;
        d_A = d_A_288_768;
    }
    if(n==288 && d==32000) {
        h_A = h_A_288_32000;
        d_A = d_A_288_32000;
    }

    float *h_B;
    float *d_B;
    if(n==288) {
        h_B = h_B_288;
        d_B = d_B_288;
    }
    if(n==768) {
        h_B = h_B_768;
        d_B = d_B_768;
    }

    float *h_C;
    float *d_C;
    if(d==288) {
        h_C = h_C_288;
        d_C = d_C_288;
    }
    if(d==768) {
        h_C = h_C_768;
        d_C = d_C_768;
    }
    if(d==32000) {
        h_C = h_C_32000;
        d_C = d_C_32000;
    }

    memcpy(h_A, w, mem_size_A);
    memcpy(h_B, x, mem_size_B);

    // copy host memory to device
    checkCudaErrors(cudaMemcpyAsync(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice, stream));

    // Setup execution parameters
    int blockSize = 32;
    dim3 threads(blockSize, blockSize);
    dim3 grid;

    grid = dim3(dimsB.x / threads.x + 1, dimsA.y / threads.y + 1);
    MatMulKernel2DAnySize<32><<<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.x, dimsC.x, dimsC.y);

    // Copy result from device to host
    checkCudaErrors(cudaMemcpyAsync(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost, stream));
    checkCudaErrors(cudaStreamSynchronize(stream));

    memcpy(xout, h_C, mem_size_C);
}
