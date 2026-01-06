#include <iostream>
#include <vector>
#include <omp.h>
#include <chrono>
#include <random>

int main(){
    const int SIZE = 1000000;

    // создаем и заполняем массив случайными числами 
    std::vector<int> arr(SIZE);
    std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count()); // Генератор случайных чисел
    std::uniform_int_distribution<int> dist(1, 10000000);// Диапазон значений

    for (int i = 0; i < SIZE; ++i) {
        arr[i] = dist(rng);// Заполняем массив
    }

    // Параллельный поиск минимума и максимума
    auto start = std::chrono::high_resolution_clock::now();

    int min_val = arr[0];
    int max_val = arr[0];

    #pragma omp parallel for reduction(min : min_val) reduction(max : max_val)
    for (int i = 0; i < SIZE; ++i) {
        if (arr[i] < min_val) {
            min_val = arr[i];
        }
        if (arr[i] > max_val) {
            max_val = arr[i];
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "Array size: " << SIZE << std::endl;
    std::cout << "Minimum value: " << min_val << std::endl;
    std::cout << "Maximum value: " << max_val << std::endl;
    std::cout << "Execution time(OpenMP): " << duration.count() << " ms" << std::endl;

    return 0;
}