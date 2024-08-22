# 设置cmake版本，如果cmake版本过高，可能会出现错误
cmake_minimum_required(VERSION 2.8.12)
project(qwen_chat)

set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-11.6)

find_package(CUDA REQUIRED)

# 设置C++编译版本
set(CMAKE_CXX_STANDARD 17)

# Pass options to NVCC
set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -g -O3 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_86,code=compute_86
    )

# Specify include directories
include_directories(
    kernels
    utility
    )

set(ABSL_ENABLE_INSTALL ON)
set(ABSL_PROPAGATE_CXX_STD ON)
add_subdirectory(third_party/abseil-cpp)
add_subdirectory(third_party/re2)

# 设置程序编译出的可执行文件
set(MAIN_FILE qwen_chat.cpp)
set(EXECUTABLE_OUTPUT_PATH ../)

add_compile_options(-g -Ofast -fopenmp)

cuda_add_executable(qwen_chat qwen_chat.cpp transformer.cu)
target_link_directories(qwen_chat PUBLIC /usr/local/cuda/lib64 /opt/cuda/lib64 $(CUDA_PATH)/targets/x86_64-linux/lib)
target_link_libraries(qwen_chat cublas culibos cudart cublasLt omp5 re2::re2)
