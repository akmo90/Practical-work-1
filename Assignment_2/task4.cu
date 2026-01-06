// Задача 4. Параллельная сортировка слиянием на GPU
// Используется CUDA
#include <iostream>
#include <cuda_runtime.h>
#include <algorithm>

using namespace std;

// Ядро сортировки внутри блока 
__global__ void blockSort(int* data, int n, int blockSize) {
    int start = blockIdx.x * blockSize;
    int end = min(start + blockSize, n);

    for (int i = start; i < end; i++) {
        for (int j = i + 1; j < end; j++) {
            if (data[j] < data[i]) {
                int tmp = data[i];
                data[i] = data[j];
                data[j] = tmp;
            }
        }
    }
}

int main() {
    const int N = 10000;   
    const int BLOCK_SIZE = 256;

    int* h_data = new int[N];
    for (int i = 0; i < N; i++)
        h_data[i] = rand() % 100000;

    int* d_data;
    cudaMalloc(&d_data, N * sizeof(int));
    cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice);

    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    blockSort<<<numBlocks, 1>>>(d_data, N, BLOCK_SIZE);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(h_data, d_data, N * sizeof(int), cudaMemcpyDeviceToHost);

    cout << "CUDA sorting time: " << ms << " ms" << endl;

    cudaFree(d_data);
    delete[] h_data;

    return 0;
}
