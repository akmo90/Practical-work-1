// Подключаем стандартные библиотеки
#include <iostream>   // для вывода в консоль
#include <cstdlib>    // для rand() и srand()
#include <ctime>      // для time()

using namespace std;

int main() {
    // Размер массива
    const int N = 50000;

    // Инициализация генератора случайных чисел
    srand(time(nullptr));

    // Динамическое выделение памяти под массив
    int* arr = new int[N];

    // Заполняем массив случайными числами от 1 до 100
    for (int i = 0; i < N; i++) {
        arr[i] = rand() % 100 + 1;
    }

    // Переменная для подсчёта суммы элементов массива
    long long sum = 0;

    // Считаем сумму всех элементов
    for (int i = 0; i < N; i++) {
        sum += arr[i];
    }

    // Вычисляем среднее значение
    double avg = static_cast<double>(sum) / N;

    // Выводим результат
    cout << "Average value: " << avg << endl;

    // Освобождаем выделенную память
    delete[] arr;

    return 0;
}
