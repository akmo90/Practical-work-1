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
        // Поэлементное сложение
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    float *d_a, *d_b, *d_c;

    // Выделение памяти на GPU
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_c, N * sizeof(float));

    // Массив с разными размерами блока потоков
    int block_sizes[] = {128, 256, 512};

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Запуск ядра для разных размеров блока
    for (int block_size : block_sizes) {
        int blocks = (N + block_size - 1) / block_size;
        float time_ms;

        cudaEventRecord(start);
        add_arrays<<<blocks, block_size>>>(d_a, d_b, d_c, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&time_ms, start, stop);

        std::cout << "Размер блока: " << block_size
                  << ", время выполнения: "
                  << time_ms << " мс" << std::endl;
    }

    // Освобождение памяти
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
