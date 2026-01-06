// Задача 2. Поиск минимума и максимума в массиве
// Последовательная и параллельная (OpenMP) версии
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <omp.h>

using namespace std;
using namespace chrono;

int main() {
    const int N = 10000;
    vector<int> arr(N);

    // Генерация случайных чисел
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(1, 100000);

    for (int i = 0; i < N; i++)
        arr[i] = dis(gen);

    // Последовательный вариант
    int min_seq = arr[0];
    int max_seq = arr[0];

    auto start_seq = high_resolution_clock::now();

    for (int i = 1; i < N; i++) {
        if (arr[i] < min_seq) min_seq = arr[i];
        if (arr[i] > max_seq) max_seq = arr[i];
    }

    auto end_seq = high_resolution_clock::now();
    auto time_seq = duration_cast<microseconds>(end_seq - start_seq).count();

    // Параллельный вариант
    int min_par = arr[0];
    int max_par = arr[0];

    auto start_par = high_resolution_clock::now();

#pragma omp parallel for reduction(min:min_par) reduction(max:max_par)
    for (int i = 0; i < N; i++) {
        if (arr[i] < min_par) min_par = arr[i];
        if (arr[i] > max_par) max_par = arr[i];
    }

    auto end_par = high_resolution_clock::now();
    auto time_par = duration_cast<microseconds>(end_par - start_par).count();

    // Вывод
    cout << "Sequential Min: " << min_seq << ", Max: " << max_seq << endl;
    cout << "Parallel   Min: " << min_par << ", Max: " << max_par << endl;

    cout << "Sequential time: " << time_seq << " microseconds" << endl;
    cout << "Parallel time:   " << time_par << " microseconds" << endl;

    return 0;
}
