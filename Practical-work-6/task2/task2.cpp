#include <CL/cl.h>
#include <iostream>
#include <vector>

// Размеры матриц
#define N 128
#define M 128
#define K 128

int main() {
    // Инициализация матриц
    std::vector<float> A(N * M, 1.0f);
    std::vector<float> B(M * K, 2.0f);
    std::vector<float> C(N * K, 0.0f);
    std::vector<float> C_cpu(N * K, 0.0f);

    // Последовательное умножение матриц на CPU (для проверки)
    for (int i = 0; i < N; i++)
        for (int j = 0; j < K; j++)
            for (int k = 0; k < M; k++)
                C_cpu[i * K + j] += A[i * M + k] * B[k * K + j];

    // Инициализация OpenCL
    cl_platform_id platform;
    cl_device_id device;
    clGetPlatformIDs(1, &platform, nullptr);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, nullptr);

    cl_context context = clCreateContext(
        nullptr, 1, &device, nullptr, nullptr, nullptr
    );
    cl_command_queue queue = clCreateCommandQueue(
        context, device, 0, nullptr
    );

    // Исходный код ядра (встроенный)
    const char* source =
        "__kernel void matrix_mul(__global float* A,__global float* B,__global float* C,int N,int M,int K){"
        "int r=get_global_id(0);int c=get_global_id(1);float s=0;"
        "for(int i=0;i<M;i++)s+=A[r*M+i]*B[i*K+c];"
        "C[r*K+c]=s;}";

    // Компиляция ядра
    cl_program program = clCreateProgramWithSource(
        context, 1, &source, nullptr, nullptr
    );
    clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    cl_kernel kernel = clCreateKernel(program, "matrix_mul", nullptr);

    // Создание буферов
    cl_mem bufA = clCreateBuffer(
        context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(float) * A.size(), A.data(), nullptr
    );
    cl_mem bufB = clCreateBuffer(
        context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(float) * B.size(), B.data(), nullptr
    );
    cl_mem bufC = clCreateBuffer(
        context, CL_MEM_WRITE_ONLY,
        sizeof(float) * C.size(), nullptr, nullptr
    );

    // Передача аргументов ядру
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC);
    clSetKernelArg(kernel, 3, sizeof(int), &N);
    clSetKernelArg(kernel, 4, sizeof(int), &M);
    clSetKernelArg(kernel, 5, sizeof(int), &K);

    // Размер глобальной рабочей области (2D)
    size_t globalSize[2] = {N, K};

    // Запуск ядра
    clEnqueueNDRangeKernel(
        queue, kernel, 2, nullptr,
        globalSize, nullptr,
        0, nullptr, nullptr
    );
    clFinish(queue);

    // Считывание результата
    clEnqueueReadBuffer(
        queue, bufC, CL_TRUE, 0,
        sizeof(float) * C.size(), C.data(),
        0, nullptr, nullptr
    );

    // Проверка корректности
    std::cout << "Result check: "
              << (C[0] == C_cpu[0] ? "OK" : "ERROR") << std::endl;

    return 0;
}
