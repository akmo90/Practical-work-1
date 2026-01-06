// установка необходимых библиотек
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <omp.h>

using namespace std;
using namespace chrono;


// сортировка пузырьком
void bubbleSort(vector<int>& a) {
    int n = a.size();
    for (int i = 0; i < n - 1; i++) // в этой сортировке сравниваем два элементы и тот что больше убираем вправо
    { 
        for (int j = 0; j < n - i - 1; j++) {
            if (a[j] > a[j + 1]) {
                swap(a[j], a[j + 1]);
            }
        }
    }
}

// сортировка выборки
void selectionSort(vector<int>& a) {
    int n = a.size();
    for (int i = 0; i < n - 1; i++) // здесь мы убираем наименьший вперед
    {
        int minIdx = i;
        for (int j = i + 1; j < n; j++) {
            if (a[j] < a[minIdx])
                minIdx = j;
        }
        swap(a[i], a[minIdx]);
    }
}

// сортировка вставкой
void insertionSort(vector<int>& a) {
    int n = a.size();
    for (int i = 1; i < n; i++) { // элементы последовательно перемещаются в отсортированный часть массива
        int key = a[i];
        int j = i - 1;
        while (j >= 0 && a[j] > key) {
            a[j + 1] = a[j];
            j--;
        }
        a[j + 1] = key;
    }
}

// параллельная реализация сортировки пузырьком с OpenMP
void bubbleSortOMP(vector<int>& a) {
    int n = a.size();
    for (int i = 0; i < n; i++) {
        #pragma omp parallel for
        for (int j = i % 2; j < n - 1; j += 2) {
            if (a[j] > a[j + 1]) {
                swap(a[j], a[j + 1]);
            }
        }
    }
}

// параллельная реализация сортировки выборки с OpenMP
void selectionSortOMP(vector<int>& a) {
    int n = a.size();
    for (int i = 0; i < n - 1; i++) {
        int minIdx = i;

        #pragma omp parallel for
        for (int j = i + 1; j < n; j++) {
            if (a[j] < a[minIdx]) {
                #pragma omp critical
                {
                    if (a[j] < a[minIdx])
                        minIdx = j;
                }
            }
        }
        swap(a[i], a[minIdx]);
    }
}

// параллельная реализация сортировки вставкой с OpenMP
void insertionSortOMP(vector<int>& a) {
    int n = a.size();
    for (int i = 1; i < n; i++) {
        int key = a[i];
        int j = i - 1;

        while (j >= 0 && a[j] > key) {
            a[j + 1] = a[j];
            j--;
        }
        a[j + 1] = key;
    }
}

// Функция генерации массива из n элементов
// Заполняет вектор случайными числами в диапазоне от 0 до 100000
vector<int> generateArray(int n) {
    vector<int> a(n);              // создаём вектор нужного размера
    random_device rd;              // источник случайности
    mt19937 gen(rd());             // генератор случайных чисел
    uniform_int_distribution<> dis(0, 100000); // диапазон значений

    // заполняем массив случайными числами
    for (int i = 0; i < n; i++)
        a[i] = dis(gen);

    return a; // возвращаем сгенерированный массив
}

// Функция тестирования алгоритма сортировки
void testSort(void (*sortFunc)(vector<int>&), vector<int> a, const string& name) {
    auto start = high_resolution_clock::now(); // начало замера времени

    sortFunc(a); // вызов переданной функции сортировки

    auto end = high_resolution_clock::now(); // конец замера времени

    // вычисляем длительность выполнения сортировки
    auto duration = duration_cast<milliseconds>(end - start);
    cout << name << ": " << duration.count() << " ms" << endl;
}

int main() {
    // Размеры массивов для тестирования
    vector<int> sizes = {1000, 10000, 100000};

    // Для каждого размера массива
    for (int size : sizes) {
        cout << "\nArray's size: " << size << endl;

        // Генерируем исходный массив случайных чисел
        vector<int> data = generateArray(size);

        // Тестируем пузырьковую сортировку
        testSort(bubbleSort, data, "Bubble Sort (seq)");
        testSort(bubbleSortOMP, data, "Bubble Sort (OMP)");

        // Тестируем сортировку выбором
        testSort(selectionSort, data, "Selection Sort (seq)");
        testSort(selectionSortOMP, data, "Selection Sort (OMP)");

        // Тестируем сортировку вставками
        testSort(insertionSort, data, "Insertion Sort (seq)");
        testSort(insertionSortOMP, data, "Insertion Sort (OMP)");
    }

    return 0; // завершение программы
}

