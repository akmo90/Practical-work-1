#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <random>             

// Функция генерации массива случайных чисел
// Используется для подготовки одинаковых входных данных
std::vector<int> generateArray(size_t n) {
    // Создаём массив заданного размера
    std::vector<int> a(n);
    // Генератор псевдослучайных чисел
    std::mt19937 gen(42);
    // Диапазон значений элементов массива
    std::uniform_int_distribution<> dis(1, 1'000'000);
    // Заполнение массива случайными значениями
    for (auto &x : a)
        x = dis(gen);

    return a;
}

// Последовательная быстрая сортировка на CPU
void cpuQuickSort(std::vector<int>&);
// Параллельная быстрая сортировка на GPU (CUDA)
void cudaQuickSort(std::vector<int>&);

// Точка входа в программу
// Выполняет сравнение производительности CPU и GPU
int main() {

    // Набор размеров массивов для тестирования
    std::vector<int> sizes = {10000, 100000, 1000000};

    // Последовательно тестируем каждый размер массива
    for (auto n : sizes) {

        // Генерируем исходный массив
        auto data = generateArray(n);

        // Создаём копии массива для CPU и GPU,
        // чтобы сравнение было корректным
        auto cpu = data;
        auto gpu = data;
        // Измерение времени выполнения сортировки на CPU
        auto start = std::chrono::high_resolution_clock::now();

        // Запуск последовательной быстрой сортировки
        cpuQuickSort(cpu);

        auto end = std::chrono::high_resolution_clock::now();

        // Вывод времени выполнения CPU
        std::cout << "CPU (" << n << " elements): "
                  << std::chrono::duration<double, std::milli>(end - start).count()
                  << " ms\n";
        // Измерение времени выполнения сортировки на GPU
        start = std::chrono::high_resolution_clock::now();

        // Запуск параллельной сортировки на GPU
        cudaQuickSort(gpu);

        end = std::chrono::high_resolution_clock::now();

        // Вывод времени выполнения GPU
        std::cout << "GPU (" << n << " elements): "
                  << std::chrono::duration<double, std::milli>(end - start).count()
                  << " ms\n\n";
    }

    // Успешное завершение программы
    return 0;
}
