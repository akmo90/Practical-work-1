#include <CL/cl.h>        // Основной заголовочный файл OpenCL
#include <iostream>      // Ввод-вывод
#include <vector>        // Контейнер vector
#include <fstream>       // Работа с файлами
#include <chrono>        // Замер времени выполнения

// Размер векторов
#define SIZE 1000000

// Функция загрузки OpenCL-ядра из файла
std::string loadKernel(const char* filename) {
    std::ifstream file(filename);
    return std::string(
        (std::istreambuf_iterator<char>(file)),
        std::istreambuf_iterator<char>()
    );
}

int main() {
    // Инициализация входных данных
    std::vector<float> A(SIZE, 1.0f);   // Вектор A
    std::vector<float> B(SIZE, 2.0f);   // Вектор B
    std::vector<float> C(SIZE);         // Результирующий вектор

    // Получение OpenCL-платформы и устройства
    cl_platform_id platform;
    cl_device_id device;
    clGetPlatformIDs(1, &platform, nullptr);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, nullptr);

    // Создание OpenCL-контекста
    cl_context context = clCreateContext(
        nullptr, 1, &device, nullptr, nullptr, nullptr
    );

    // Создание очереди команд
    cl_command_queue queue = clCreateCommandQueue(
        context, device, 0, nullptr
    );

    // Загрузка и компиляция OpenCL-ядра
    std::string source = loadKernel("kernel_vector_add.cl");
    const char* src = source.c_str();
    size_t length = source.size();

    cl_program program = clCreateProgramWithSource(
        context, 1, &src, &length, nullptr
    );
    clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);

    // Создание объекта ядра
    cl_kernel kernel = clCreateKernel(program, "vector_add", nullptr);

    // Создание буферов в глобальной памяти устройства
    cl_mem bufA = clCreateBuffer(
        context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(float) * SIZE, A.data(), nullptr
    );
    cl_mem bufB = clCreateBuffer(
        context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(float) * SIZE, B.data(), nullptr
    );
    cl_mem bufC = clCreateBuffer(
        context, CL_MEM_WRITE_ONLY,
        sizeof(float) * SIZE, nullptr, nullptr
    );

    // Передача аргументов в ядро
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC);

    // Размер глобальной рабочей области
    size_t globalSize = SIZE;

    // Замер времени выполнения ядра
    auto start = std::chrono::high_resolution_clock::now();

    // Запуск OpenCL-ядра
    clEnqueueNDRangeKernel(
        queue, kernel, 1, nullptr,
        &globalSize, nullptr,
        0, nullptr, nullptr
    );

    // Ожидание завершения всех команд
    clFinish(queue);

    auto end = std::chrono::high_resolution_clock::now();

    // Копирование результата из устройства в хост-память
    clEnqueueReadBuffer(
        queue, bufC, CL_TRUE, 0,
        sizeof(float) * SIZE, C.data(),
        0, nullptr, nullptr
    );

    // Вывод времени выполнения
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Execution time: " << elapsed.count() << " seconds\n";

    // Освобождение ресурсов OpenCL
    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufC);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
