#include <iostream>              // Ввод и вывод в консоль
#include <vector>                // Контейнер std::vector
#include <chrono>                // Измерение времени выполнения
#include <random>                // Генерация случайных чисел
#include <cuda_runtime.h>        // Основные функции CUDA

// Функция генерации массива случайных чисел на CPU
// Используется для подготовки входных данных
std::vector<int> generateArray(size_t n) {

    // Создаём вектор заданного размера
    std::vector<int> a(n);
    std::mt19937 gen(42);

    // Диапазон случайных значений
    std::uniform_int_distribution<> dis(1, 1'000'000);

    // Заполняем массив случайными числами
    for (auto &x : a)
        x = dis(gen);

    // Возвращаем заполненный массив
    return a;
}

// CUDA сортировки блока
__global__ void blockSort(int* data, int n, int blockSize) {

    // Вычисляем начальный индекс подмассива
    int start = blockIdx.x * blockSize;

    // Вычисляем конечный индекс подмассива
    // Используем явную проверку вместо min для совместимости
    int end = (start + blockSize < n) ? (start + blockSize) : n;

    // Последовательная сортировка внутри подмассива
    for (int i = start; i < end; ++i) {
        for (int j = i + 1; j < end; ++j) {

            // Меняем элементы местами, если они неупорядочены
            if (data[j] < data[i]) {
                int tmp = data[i];
                data[i] = data[j];
                data[j] = tmp;
            }
        }
    }
}

// Сортировка слиянием
// Каждый блок сортирует свой участок массива параллельно
void cudaMergeSort(std::vector<int>& h) {

    int* d;                 // Указатель на память GPU
    int n = h.size();       // Размер массива

    // Выделяем память на видеокарте
    cudaMalloc(&d, n * sizeof(int));

    // Копируем данные с CPU на GPU
    cudaMemcpy(d, h.data(),
               n * sizeof(int),
               cudaMemcpyHostToDevice);

    // Размер подмассива который обрабатывается одним блоком
    int blockSize = 256;
    // Количество блоков
    int blocks = (n + blockSize - 1) / blockSize;
    // Запуск CUDA-ядра
    blockSort<<<blocks, 1>>>(d, n, blockSize);
    // Ожидаем завершения всех GPU-потоков
    cudaDeviceSynchronize();
    // Копируем отсортированные данные обратно на CPU
    cudaMemcpy(h.data(),
               d,
               n * sizeof(int),
               cudaMemcpyDeviceToHost);
    // Освобождаем память GPU
    cudaFree(d);
}

// Точка входа в программу
int main() {
    // Генерируем массив из 100 000 элементов
    auto data = generateArray(100000);
    // Засекаем время начала выполнения
    auto start = std::chrono::high_resolution_clock::now();
    // Запуск CUDA-сортировки
    cudaMergeSort(data);
    // Засекаем время окончания выполнения
    auto end = std::chrono::high_resolution_clock::now();
    // Вывод времени выполнения сортировки
    std::cout << "CUDA Merge Sort time: "
              << std::chrono::duration<double, std::milli>(end - start).count()
              << " ms\n";

    return 0;
}
