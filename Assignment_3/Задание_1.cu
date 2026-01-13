#include <cuda_runtime.h>
#include <iostream>

// Размер массива
#define N 1000000

// Размер блока потоков
#define BLOCK_SIZE 256

// поэлементное умножение с использованием только глобальной памяти
__global__ void multiply_global(float* data, float factor, int n) {
    // Глобальный индекс потока
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Проверка выхода за границы массива
    if (idx < n) {
        // Умножаем элемент массива на число
        data[idx] *= factor;
    }
}

// поэлементное умножение с использованием разделяемой (shared) памяти

__global__ void multiply_shared(float* data, float factor, int n) {
    // Разделяемая память внутри блока
    __shared__ float temp[BLOCK_SIZE];

    // Глобальный индекс потока
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Локальный индекс потока в блоке
    int tid = threadIdx.x;

    // Проверка выхода за границы массива
    if (idx < n) {
        // Копирование данных из глобальной памяти в разделяемую память
        temp[tid] = data[idx];
        __syncthreads();

        // Умножение элемента в разделяемой памяти
        temp[tid] *= factor;
        __syncthreads();

        // Запись результата обратно в глобальную память
        data[idx] = temp[tid];
    }
}

int main() {
    float* d_data;

    // Выделение памяти на GPU
    cudaMalloc(&d_data, N * sizeof(float));

    // Вычисление количества блоков
    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // CUDA для замера времени
    cudaEvent_t start, stop;
    float time_global, time_shared;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Замер времени выполнения ядра с глобальной памятью
    cudaEventRecord(start);
    multiply_global<<<blocks, BLOCK_SIZE>>>(d_data, 2.0f, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_global, start, stop);

    // Замер времени выполнения ядра с разделяемой памятью
    cudaEventRecord(start);
    multiply_shared<<<blocks, BLOCK_SIZE>>>(d_data, 2.0f, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_shared, start, stop);

    // Вывод результатов
    std::cout << "Размер массива: " << N << std::endl;
    std::cout << "Время (глобальная память): " 
              << time_global << " мс" << std::endl;
    std::cout << "Время (разделяемая память): " 
              << time_shared << " мс" << std::endl;

    // Освобождение ресурсов
    cudaFree(d_data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
