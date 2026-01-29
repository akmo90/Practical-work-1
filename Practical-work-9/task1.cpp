#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>

using namespace std;

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);   // Инициализация MPI

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // номер процесса
    MPI_Comm_size(MPI_COMM_WORLD, &size); // всего процессов

    const int N = 1000000;    // размер массива

    vector<double> data;     // общий массив (только у rank 0)
    vector<int> counts(size), displs(size);

    // Расчёт размеров частей для Scatterv
    int base = N / size;
    int rem  = N % size;

    for (int i = 0; i < size; i++)
    {
        counts[i] = base + (i < rem ? 1 : 0);
        displs[i] = (i == 0) ? 0 : displs[i-1] + counts[i-1];
    }

    if (rank == 0)
    {
        data.resize(N);
        for (int i = 0; i < N; i++)
            data[i] = rand() / (double)RAND_MAX;
    }

    // Локальный буфер
    vector<double> local_data(counts[rank]);

    double start_time = MPI_Wtime();

    // Распределение данных
    MPI_Scatterv(
        data.data(), counts.data(), displs.data(), MPI_DOUBLE,
        local_data.data(), counts[rank], MPI_DOUBLE,
        0, MPI_COMM_WORLD
    );

    // Локальные суммы
    double local_sum = 0;
    double local_sq_sum = 0;

    for (double x : local_data)
    {
        local_sum += x;
        local_sq_sum += x * x;
    }

    // Глобальные суммы
    double global_sum = 0;
    double global_sq_sum = 0;

    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_sq_sum, &global_sq_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    double end_time = MPI_Wtime();

    if (rank == 0)
    {
        double mean = global_sum / N;
        double variance = global_sq_sum / N - mean * mean;
        double stddev = sqrt(variance);

        cout << "Processes: " << size << endl;
        cout << "Mean: " << mean << endl;
        cout << "Std deviation: " << stddev << endl;
        cout << "Execution time: " << end_time - start_time << " seconds" << endl;
    }

    MPI_Finalize(); // Завершение MPI
    return 0;
}
