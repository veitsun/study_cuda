#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>

using namespace std;
// 向量乘法内积形式

__global__ void VecMatrix(float *A, float *B, float *Num, int N) {
  extern __shared__ float sharedMem[]; // extern 表示该变量是在其他地方定义的，
                                       // 当前声明只是告诉编译器有这个变量存在
  // __shared__
  // 表示该变量是块级别的共享内存，即同一块内的所有线程都可以访问这块内存
  // 共享内存是CUDA中的内存类型，它位于GPU上，同一块内的所有线程都可以访问这块内存
  int i = threadIdx.x;

  if (i < N)
    sharedMem[i] = A[i] * B[i];
  __syncthreads(); // 等待所有线程完成计算

  if (threadIdx.x == 0) {
    // for ()
    *Num = 0;
    for (int j = 0; j < N; j++) {
      *Num += sharedMem[j];
    }
    printf("Device Num :%f\n", *Num);
  }

  __syncthreads();
}

int main() {
  const int N = 5;
  // 这是主机上的数据
  float HostA[] = {2.0, 2.0, 2.0, 2.0, 2.0};
  float HostB[] = {3.0, 4.0, 5.0, 6.0, 7.0};
  float HostNum = 0.5;

  // 在设备上分配GPU内存
  float *deviceA, *deviceB, *deviceNum;
  cudaMalloc(&deviceA, N * sizeof(float));
  cudaMalloc(&deviceB, N * sizeof(float));
  cudaMalloc(&deviceNum, sizeof(float));

  // 将主机上的数据复制到设备上
  cudaMemcpy(deviceA, HostA, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, HostB, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceNum, &HostNum, sizeof(float), cudaMemcpyHostToDevice);

  // 执行内核函数
  VecMatrix<<<1, N>>>(deviceA, deviceB, deviceNum, N);

  // 将设备上的数据复制到主机上
  cudaMemcpy(HostA, deviceA, N * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(HostB, deviceB, N * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(&HostNum, deviceNum, sizeof(float), cudaMemcpyDeviceToHost);

  // 从主机上输出A
  cout << "printf A" << endl;
  for (int i = 0; i < N; i++) {
    cout << HostA[i] << " ";
  }
  cout << endl;

  // 从主机上输出B
  cout << "pprintf B " << endl;
  for (int i = 0; i < N; i++) {
    cout << HostB[i] << " ";
  }
  cout << endl;

  // 从主机上输出结果
  cout << HostNum << endl;

  return 0;
}