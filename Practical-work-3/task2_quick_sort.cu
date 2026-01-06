#include <iostream>              // вывод в консоль
#include <vector>                // контейнер vector
#include <chrono>                // измерение времени
#include <random>                // генерация случайных чисел
#include <cuda_runtime.h>        // CUDA API

// Функция генерации массива случайных чисел (на CPU)
std::vector<int> generateArray(size_t n) {
    std::vector<int> a(n);

    // Фиксированное зерно — одинаковые данные при каждом запуске
    std::mt19937 gen(42);
    std::uniform_int_distribution<> dis(1, 1'000'000);

    for (auto &x : a)
        x = dis(gen);

    return a;
}

// Рекурсивная быстрая сортировка 
// __device__ означает, что функция вызывается только с GPU
__device__ void quickSortDevice(int* arr, int left, int right) {
    int i = left;
    int j = right;

    // Опорный элемент (pivot)
    int pivot = arr[(left + right) / 2];

    // Основной цикл разбиения
    while (i <= j) {
        while (arr[i] < pivot) i++;
        while (arr[j] > pivot) j--;

        if (i <= j) {
            int tmp = arr[i];
            arr[i] = arr[j];
            arr[j] = tmp;
            i++;
            j--;
        }
    }

    // Рекурсивная сортировка подмассивов
    if (left < j)
        quickSortDevice(arr, left, j);
    if (i < right)
        quickSortDevice(arr, i, right);
}
// Каждый поток сортирует свой участок массива
__global__ void quickSortKernel(int* data, int n, int chunkSize) {

    // Глобальный ID потока
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Начальный индекс участка
    int start = tid * chunkSize;

    // Конечный индекс (без использования min!)
    int end = (start + chunkSize - 1 < n - 1)
                ? (start + chunkSize - 1)
                : (n - 1);

    // Если участок корректный — сортируем
    if (start < end) {
        quickSortDevice(data, start, end);
    }
}
// Обёртка CUDA-сортировки (вызывается с CPU)
void cudaQuickSort(std::vector<int>& hostData) {

    int n = hostData.size();
    int* deviceData;

    // Выделение памяти на GPU
    cudaMalloc(&deviceData, n * sizeof(int));

    // Копирование данных CPU → GPU
    cudaMemcpy(deviceData, hostData.data(),
               n * sizeof(int), cudaMemcpyHostToDevice);

    // Конфигурация запуска
    int threadsPerBlock = 256;
    int blocks = 256;

    // Размер участка для одного потока
    int chunkSize = n / (threadsPerBlock * blocks) + 1;

    // Запуск CUDA-ядра
    quickSortKernel<<<blocks, threadsPerBlock>>>(
        deviceData, n, chunkSize
    );

    // Ожидание завершения всех потоков
    cudaDeviceSynchronize();

    // Копирование результата GPU → CPU
    cudaMemcpy(hostData.data(), deviceData,
               n * sizeof(int), cudaMemcpyDeviceToHost);

    // Освобождение памяти GPU
    cudaFree(deviceData);
}

// Точка входа в программу
int main() {

    // Генерация массива
    auto data = generateArray(100000);

    // Замер времени
    auto start = std::chrono::high_resolution_clock::now();
    cudaQuickSort(data);
    auto end = std::chrono::high_resolution_clock::now();

    // Вывод времени выполнения
    std::cout << "CUDA Quick Sort time: "
              << std::chrono::duration<double, std::milli>(end - start).count()
              << " ms\n";

    return 0;
}
