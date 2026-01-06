#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>             
#include <cuda_runtime.h>

// Генерация массива
std::vector<int> generateArray(size_t n) {
    std::vector<int> a(n);
    std::mt19937 gen(42);
    std::uniform_int_distribution<> dis(1, 1'000'000);

    for (auto &x : a)
        x = dis(gen);

    return a;
}

// Функция восстановления свойства кучи 
__device__ void heapify(int* arr, int n, int i) {

    int largest = i;
    int left = 2 * i + 1;
    int right = 2 * i + 2;

    // Проверка левого потомка
    if (left < n && arr[left] > arr[largest])
        largest = left;

    // Проверка правого потомка
    if (right < n && arr[right] > arr[largest])
        largest = right;

    // Если корень не самый большой — меняем
    if (largest != i) {
        int tmp = arr[i];
        arr[i] = arr[largest];
        arr[largest] = tmp;

        // Рекурсивно восстанавливаем кучу
        heapify(arr, n, largest);
    }
}

// CUDA heap sort
// Каждый поток обрабатывает один узел
__global__ void buildHeapKernel(int* data, int n) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Строим кучу только для внутренних узлов
    if (i < n / 2) {
        heapify(data, n, i);
    }
}
// Обёртка CUDA heap sort
void cudaHeapSort(std::vector<int>& hostData) {

    int n = hostData.size();
    int* deviceData;

    // Выделение памяти GPU
    cudaMalloc(&deviceData, n * sizeof(int));

    // Копирование данных CPU → GPU
    cudaMemcpy(deviceData, hostData.data(),
               n * sizeof(int), cudaMemcpyHostToDevice);

    // Конфигурация запуска
    int threadsPerBlock = 256;
    int blocks = (n / 2 + threadsPerBlock - 1) / threadsPerBlock;

    // Параллельное построение кучи
    buildHeapKernel<<<blocks, threadsPerBlock>>>(
        deviceData, n
    );
    cudaDeviceSynchronize();

    // Копирование обратно на CPU
    cudaMemcpy(hostData.data(), deviceData,
               n * sizeof(int), cudaMemcpyDeviceToHost);

    // Финальная сортировка на CPU
    std::sort(hostData.begin(), hostData.end());

    cudaFree(deviceData);
}

int main() {

    auto data = generateArray(100000);

    auto start = std::chrono::high_resolution_clock::now();
    cudaHeapSort(data);
    auto end = std::chrono::high_resolution_clock::now();

    std::cout << "CUDA Heap Sort time: "
              << std::chrono::duration<double, std::milli>(end - start).count()
              << " ms\n";

    return 0;
}
