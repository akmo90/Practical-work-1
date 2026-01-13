#include <cuda_runtime.h>
#include <iostream>

// Размер массива
#define N 1000000

// Размер блока потоков
#define BLOCK_SIZE 256

// Ядро с коалесцированным доступом к глобальной памяти
// Потоки одного warp обращаются к соседним элементам
__global__ void coalesced_access(float* data, int n) {
    // Глобальный индекс потока
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Проверка выхода за границы массива
    if (idx < n) {
        // Последовательный доступ к памяти
        data[idx] *= 2.0f;
    }
}

// Ядро с некоалесцированным доступом к глобальной памяти
// Потоки обращаются к памяти с большим шагом
__global__ void uncoalesced_access(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Искусственно создаём разреженный доступ
    int access_idx = (idx * 32) % n;

    if (access_idx < n) {
        data[access_idx] *= 2.0f;
    }
}

int main() {
    float* d_data;

    // Выделяем память на GPU
    cudaMalloc(&d_data, N * sizeof(float));

    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cudaEvent_t start, stop;
    float time_coalesced, time_uncoalesced;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Замер времени для коалесцированного доступа
    cudaEventRecord(start);
    coalesced_access<<<blocks, BLOCK_SIZE>>>(d_data, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_coalesced, start, stop);

    // Замер времени для некоалесцированного доступа
    cudaEventRecord(start);
    uncoalesced_access<<<blocks, BLOCK_SIZE>>>(d_data, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_uncoalesced, start, stop);

    // Вывод результатов
    std::cout << "Размер массива: " << N << std::endl;
    std::cout << "Коалесцированный доступ: "
              << time_coalesced << " мс" << std::endl;
    std::cout << "Некоалесцированный доступ: "
              << time_uncoalesced << " мс" << std::endl;

    // Освобождение ресурсов
    cudaFree(d_data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
