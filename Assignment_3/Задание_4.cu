#include <cuda_runtime.h>
#include <iostream>

// Размер массивов
#define N 1000000

// CUDA-ядро: поэлементное сложение двух массивов
__global__ void add_arrays(float* a, float* b, float* c, int n) {
    // Глобальный индекс потока
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Проверка выхода за границы массива
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    float *d_a, *d_b, *d_c;

    // Выделение памяти на GPU
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_c, N * sizeof(float));

    // CUDA события для замера времени
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float time_bad, time_good;

    // Неоптимальная конфигурация (малый размер блока)
    int bad_block_size = 64;
    int bad_blocks = (N + bad_block_size - 1) / bad_block_size;

    cudaEventRecord(start);
    add_arrays<<<bad_blocks, bad_block_size>>>(d_a, d_b, d_c, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_bad, start, stop);

    // Оптимизированная конфигурация
    int good_block_size = 256;
    int good_blocks = (N + good_block_size - 1) / good_block_size;

    cudaEventRecord(start);
    add_arrays<<<good_blocks, good_block_size>>>(d_a, d_b, d_c, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_good, start, stop);

    // Вывод результатов
    std::cout << "Размер массива: " << N << std::endl;
    std::cout << "Неоптимальная конфигурация (block = 64): "
              << time_bad << " мс" << std::endl;
    std::cout << "Оптимизированная конфигурация (block = 256): "
              << time_good << " мс" << std::endl;

    // Освобождение ресурсов
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
