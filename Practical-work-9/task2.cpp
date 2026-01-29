#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int N = 4; // размер системы (можно менять)
    
    vector<double> A; // матрица (только у rank 0)
    vector<double> b; // правая часть
    vector<double> x(N); // решение

    // Локальные строки
    int rows_per_proc = N / size;
    if (N % size != 0 && rank == size - 1)
        rows_per_proc += N % size;

    vector<double> local_A(rows_per_proc * N);
    vector<double> local_b(rows_per_proc);

    if (rank == 0)
    {
        // Пример системы:
        // 2x +  y -  z =  8
        //-3x -  y + 2z = -11
        //-2x +  y + 2z = -3
        // x + 2y + 3z = 13

        A = {
             2,  1, -1,  1,
            -3, -1,  2,  2,
            -2,  1,  2,  3,
             1,  2,  3,  4
        };

        b = {8, -11, -3, 13};

        cout << "Исходная система:\n";
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
                cout << A[i*N + j] << " ";
            cout << "| " << b[i] << endl;
        }
    }

    double start_time = MPI_Wtime();

    // Распределяем строки матрицы
    MPI_Scatter(
        A.data(), rows_per_proc * N, MPI_DOUBLE,
        local_A.data(), rows_per_proc * N, MPI_DOUBLE,
        0, MPI_COMM_WORLD
    );

    MPI_Scatter(
        b.data(), rows_per_proc, MPI_DOUBLE,
        local_b.data(), rows_per_proc, MPI_DOUBLE,
        0, MPI_COMM_WORLD
    );

    vector<double> pivot_row(N);
    double pivot_b;

    // -------- Прямой ход --------
    for (int k = 0; k < N; k++)
    {
        int owner = k / rows_per_proc;

        if (rank == owner)
        {
            int local_k = k % rows_per_proc;
            for (int j = 0; j < N; j++)
                pivot_row[j] = local_A[local_k * N + j];
            pivot_b = local_b[local_k];
        }

        // Рассылка ведущей строки
        MPI_Bcast(pivot_row.data(), N, MPI_DOUBLE, owner, MPI_COMM_WORLD);
        MPI_Bcast(&pivot_b, 1, MPI_DOUBLE, owner, MPI_COMM_WORLD);

        // Обновление локальных строк
        for (int i = 0; i < rows_per_proc; i++)
        {
            int global_i = rank * rows_per_proc + i;
            if (global_i > k)
            {
                double factor = local_A[i*N + k] / pivot_row[k];
                for (int j = k; j < N; j++)
                    local_A[i*N + j] -= factor * pivot_row[j];
                local_b[i] -= factor * pivot_b;
            }
        }
    }

    // Сбор матрицы обратно
    MPI_Gather(
        local_A.data(), rows_per_proc * N, MPI_DOUBLE,
        A.data(), rows_per_proc * N, MPI_DOUBLE,
        0, MPI_COMM_WORLD
    );

    MPI_Gather(
        local_b.data(), rows_per_proc, MPI_DOUBLE,
        b.data(), rows_per_proc, MPI_DOUBLE,
        0, MPI_COMM_WORLD
    );

    // -------- Обратный ход (только rank 0) --------
    if (rank == 0)
    {
        for (int i = N - 1; i >= 0; i--)
        {
            x[i] = b[i];
            for (int j = i + 1; j < N; j++)
                x[i] -= A[i*N + j] * x[j];
            x[i] /= A[i*N + i];
        }
    }

    double end_time = MPI_Wtime();

    if (rank == 0)
    {
        cout << "\nРешение системы:\n";
        for (int i = 0; i < N; i++)
            cout << "x[" << i << "] = " << x[i] << endl;

        cout << "\nExecution time: "
             << end_time - start_time
             << " seconds\n";
    }

    MPI_Finalize();
    return 0;
}
