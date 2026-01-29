// OpenCL-ядро для умножения матриц A(NxM) и B(MxK)
// Результат сохраняется в матрицу C(NxK)
__kernel void matrix_mul(
    __global float* A,  // Матрица A
    __global float* B,  // Матрица B
    __global float* C,  // Результирующая матрица C
    int N,              // Количество строк A
    int M,              // Количество столбцов A / строк B
    int K               // Количество столбцов B
) {
    // Индексы строки и столбца
    int row = get_global_id(0);
    int col = get_global_id(1);

    float sum = 0.0f;

    // Скалярное произведение строки и столбца
    for (int i = 0; i < M; i++) {
        sum += A[row * M + i] * B[i * K + col];
    }

    // Запись результата
    C[row * K + col] = sum;
}
