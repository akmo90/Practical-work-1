#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <omp.h>

int main() {
    // Размер массива
    const int SIZE = 5000000;

    // Создаем массив и заполняем его случайными числами
    std::vector<int> arr(SIZE);
    std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());
    std::uniform_int_distribution<int> dist(1, 100);

    for (int i = 0; i < SIZE; ++i) {
        arr[i] = dist(rng);
    }

    //последовательское вычисление
    auto start_seq = std::chrono::high_resolution_clock::now();

    long long sum_seq = 0;
    for (int i = 0; i < SIZE; ++i) {
        sum_seq += arr[i];
    }
    double avg_seq = static_cast<double>(sum_seq) / SIZE;

    auto end_seq = std::chrono::high_resolution_clock::now();
    auto time_seq =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_seq - start_seq);

    // параллельное вычисление OpenMP
    auto start_par = std::chrono::high_resolution_clock::now();

    long long sum_par = 0;
#pragma omp parallel for reduction(+ : sum_par)
    for (int i = 0; i < SIZE; ++i) {
        sum_par += arr[i];
    }
    double avg_par = static_cast<double>(sum_par) / SIZE;

    auto end_par = std::chrono::high_resolution_clock::now();
    auto time_par =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_par - start_par);// время выполнения алгоритма в миллисекундах

    // вывод результатов
    std::cout << "Array size: " << SIZE << std::endl;// выводим размер массива

    std::cout << "Sequential average: " << avg_seq
              << ", time: " << time_seq.count() << " ms" << std::endl;//выводим среднее значение последовательным способом

    std::cout << "Parallel average (OpenMP): " << avg_par
              << ", time: " << time_par.count() << " ms" << std::endl;// выводим среднее значение параллельным способом openmp

    return 0;
}
