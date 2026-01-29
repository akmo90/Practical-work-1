#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <algorithm>

using namespace std;

#define INF 1e9

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int N = 6; // размер графа (можно менять)

    vector<double> G; // матрица смежности (только у rank 0)

    int rows_per_proc = N / size;
    if (N % size != 0 && rank == size - 1)
        rows_per_proc += N % size;

    vector<double> local_G(rows_per_proc * N);

    if (rank == 0)
    {
        // Пример графа (матрица смежности)
        // INF означает отсутствие ребра
        G = {
            0,   3, INF,   7, INF, INF,
            8,   0,   2, INF, INF, INF,
            5, INF,   0,   1, INF, INF,
            2, INF, INF,   0,   1, INF,
          INF, INF, INF, INF,   0,   2,
          INF, INF, INF, INF, INF,   0
        };

        cout << "Исходная матрица графа:\n";
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                if (G[i*N + j] >= INF)
                    cout << "INF ";
                else
                    cout << G[i*N + j] << " ";
            }
            cout << endl;
        }
    }

    double start_time = MPI_Wtime();

    // Распределяем строки матрицы
    MPI_Scatter(
        G.data(), rows_per_proc * N, MPI_DOUBLE,
        local_G.data(), rows_per_proc * N, MPI_DOUBLE,
        0, MPI_COMM_WORLD
    );

    // Глобальная матрица для синхронизации
    vector<double> global_G(N * N);

    // -------- Алгоритм Флойда–Уоршелла --------
    for (int k = 0; k < N; k++)
    {
        // Собираем актуальную матрицу со всех процессов
        MPI_Allgather(
            local_G.data(), rows_per_proc * N, MPI_DOUBLE,
            global_G.data(), rows_per_proc * N, MPI_DOUBLE,
            MPI_COMM_WORLD
        );

        // Обновляем локальные строки
        for (int i = 0; i < rows_per_proc; i++)
        {
            int global_i = rank * rows_per_proc + i;
            for (int j = 0; j < N; j++)
            {
                double through_k =
                    global_G[global_i*N + k] +
                    global_G[k*N + j];

                local_G[i*N + j] =
                    min(local_G[i*N + j], through_k);
            }
        }
    }

    // Собираем финальную матрицу
    MPI_Gather(
        local_G.data(), rows_per_proc * N, MPI_DOUBLE,
        G.data(), rows_per_proc * N, MPI_DOUBLE,
        0, MPI_COMM_WORLD
    );

    double end_time = MPI_Wtime();

    if (rank == 0)
    {
        cout << "\nМатрица кратчайших путей:\n";
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                if (G[i*N + j] >= INF)
                    cout << "INF ";
                else
                    cout << G[i*N + j] << " ";
            }
            cout << endl;
        }

        cout << "\nExecution time: "
             << end_time - start_time
             << " seconds\n";
    }

    MPI_Finalize();
    return 0;
}
