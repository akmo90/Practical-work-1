#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <random>

int main(){
    //ставим константу из 1000000
    const int SIZE= 1000000;

    // создаем и заполняем массив случайными числами 
    std::vector<int> arr(SIZE);
    std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count()); // Генератор случайных чисел
    std::uniform_int_distribution<int> dist(1, 10000000); // Диапазон значений

    for (int i = 0; i < SIZE; ++i) {
        arr[i] = dist(rng); // Заполняем массив
    }

    // последовательный алгоритм для поиска минимального и максимального
    auto start = std::chrono::high_resolution_clock::now();

    int min_val = arr[0]; // Предполагаем, что у нас первый элемент минимум
    int max_val = arr[0]; // Здесь у нас может быть первый элемент максимум

    for(int i=1; i<SIZE; ++i){
        if(arr[i]<min_val){
            min_val = arr[i]; //находим новый минимум
        }
        if(arr[i]>max_val){
            max_val = arr[i]; // находим новый максимум
        }
    }

    // здесь заканчивается замер времени 
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start); // время выполнения алгоритма

    std::cout << "Massive size: " << SIZE << std::endl;
    std::cout << "Minimal value: " << min_val << std::endl;
    std::cout << "Maximum value: " << max_val << std::endl;
    std::cout << "Time for algorithm: " << duration.count() << " ms" << std::endl;
}