// task3_selection_sort.cpp
// =====================================================
// Задача 3. Сортировка выбором
// Последовательная и параллельная (OpenMP) реализации
// =====================================================

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <omp.h>

using namespace std;
using namespace chrono;

// Последовательная сортировка выбором
void selectionSortSeq(vector<int>& a) {
    int n = a.size();
    for (int i = 0; i < n - 1; i++) {
        int min_idx = i;
        for (int j = i + 1; j < n; j++) {
            if (a[j] < a[min_idx])
                min_idx = j;
        }
        swap(a[i], a[min_idx]);
    }
}

// Параллельная сортировка выбором
void selectionSortOMP(vector<int>& a) {
    int n = a.size();

    for (int i = 0; i < n - 1; i++) {
        int min_idx = i;

#pragma omp parallel for
        for (int j = i + 1; j < n; j++) {
#pragma omp critical
            {
                if (a[j] < a[min_idx])
                    min_idx = j;
            }
        }
        swap(a[i], a[min_idx]);
    }
}

vector<int> generateArray(int n) {
    vector<int> a(n);
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(1, 100000);

    for (int i = 0; i < n; i++)
        a[i] = dis(gen);

    return a;
}

int main() {
    vector<int> sizes = {1000, 10000};

    for (int size : sizes) {
        cout << "\nArray size: " << size << endl;

        vector<int> a1 = generateArray(size);
        vector<int> a2 = a1;

        auto start_seq = high_resolution_clock::now();
        selectionSortSeq(a1);
        auto end_seq = high_resolution_clock::now();

        auto start_par = high_resolution_clock::now();
        selectionSortOMP(a2);
        auto end_par = high_resolution_clock::now();

        cout << "Sequential time: "
             << duration_cast<milliseconds>(end_seq - start_seq).count()
             << " ms" << endl;

        cout << "Parallel time:   "
             << duration_cast<milliseconds>(end_par - start_par).count()
             << " ms" << endl;
    }

    return 0;
}
