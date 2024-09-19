// kernel definition
#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>


__global__ void VecAdd(float *A, float *B, float *C, int N) {
    // 每个执行内核的线程都有一个唯一的线程 ID，可以通过内置变量在内核中访问
    int i = threadIdx.x;
    if(i < N){
        C[i] = A[i] + B[i];
        // 这里只是为了调试，通常不建议在内核中使用printf
        printf("%f\n", C[i]);
        //std::cout << "sunwei" << std::endl;
    }
}


int main()
{
    const int N = 5;

    // 下面的ABC是主机上的数据
    float A[] = {1.0, 1.0, 1.0, 1.0, 1.0};
    float B[] = {1.0, 1.0, 1.0, 1.0, 1.0};
    float C[] = {0.0, 0.0, 0.0, 0.0, 0.0};


    // 分配设备内存
    float *deviceA, *deviceB, *deviceC;
    cudaMalloc(&deviceA, N * sizeof(float));
    cudaMalloc(&deviceB, N * sizeof(float));
    cudaMalloc(&deviceC, N * sizeof(float));


    // 将数据从主机复制到设备
    cudaMemcpy((void **)deviceA, A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy((void **)deviceB, B, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy((void **)deviceC, C, N * sizeof(float), cudaMemcpyHostToDevice);




    // 执行内核函数
    VecAdd<<<1, 5>>>(deviceA, deviceB, deviceC, N);

    // 检查是否有cuda错误
    

    // 将结果从设备复制回主机
    cudaMemcpy(C, deviceC, N * sizeof(float), cudaMemcpyDeviceToHost);


    // 输出结果
    std::cout << "Result C:";
    for(int i = 0; i < N; i ++) {
        std::cout << C[i] << " ";
    }
    std::cout << std::endl;

    // 释放设备内存
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);

    return 0;
}

