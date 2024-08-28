# 设置cmake版本，如果cmake版本过高，可能会出现错误
cmake_minimum_required(VERSION 2.8.12)
project(run_onnx)

# 设置C++编译版本
set(CMAKE_CXX_STANDARD 17)

# Specify include directories
include_directories(
    onnxruntime-linux-x64-1.19.0/include
)

# 设置程序编译出的可执行文件
set(MAIN_FILE run_onnx.cpp)
set(EXECUTABLE_OUTPUT_PATH ../)

add_compile_options(-g -Ofast -fopenmp)

add_executable(run_onnx run_onnx.cpp)
target_link_directories(run_onnx PUBLIC onnxruntime-linux-x64-1.19.0/lib)
target_link_libraries(run_onnx omp5 onnxruntime)
