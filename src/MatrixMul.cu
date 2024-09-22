#include "../include/CGemmWithC.h"
#include "../include/common.h"
#include "../include/mycuda.h"
#include <cstdio>
#include <cstdlib>

using namespace std;

void checkResult(float *hostRef, float *gpuRef, const int N) {
  double epsilon = 1.0E-8;
  bool match = 1;

  for (int i = 0; i < N; i++) {
    if (abs(hostRef[i] - gpuRef[i]) > epsilon) {
      match = 0;
      printf("Arrays do not match!\n");
      printf("%d\n", i);
      printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i], gpuRef[i], i);
      break;
    }
  }

  if (match)
    printf("Arrays match.\n\n");

  return;
}

void initialData(float *ip, int size) {
  // generate different seed for random number
  time_t t;
  srand((unsigned)time(&t));

  for (int i = 0; i < size; i++) {
    ip[i] = (float)(rand() & 0xFF) / 10.0f;
  }

  return;
}

void printMatrix(float *matrix, int size, int nx, int ny) {
  printf("Matrix: \n");
  float *A = matrix;
  for (int i = 0; i < size; i++) {

    printf("%f ", *(A + i));
    if ((i + 1) % ny == 0) {
      printf("\n");
    }
  }
  printf("\n\n");
}

// __global__ void MulMatrixOnDevice(float *A, float *B, float *C, int nx,
//                                   int ny) {}

__global__ void MulMatrixOnDevice(int M, int N, int K, float alpha, float *A,
                                  float *B, float beta, float *C) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < M && col < N) {
    float temp = 0.0;
    for (int k = 0; k < K; k++) {
      temp += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = alpha * temp + beta * C[row * N + col];
  }
}

int main(int argc, char **argv) {
  float *hostA;
  float *hostB;
  float *hostC;
  float *hostRef;
  float *gpuRef;

  int nx = 8;
  int ny = 8;
  int elemNum = nx * ny;

  // 给主机上的三个矩阵分配内存
  hostA = (float *)malloc(elemNum * sizeof(float));
  hostB = (float *)malloc(elemNum * sizeof(float));
  hostC = (float *)malloc(elemNum * sizeof(float));
  hostRef = (float *)malloc(elemNum * sizeof(float));
  gpuRef = (float *)malloc(elemNum * sizeof(float));

  // 主机上的三个矩阵初始化数据
  initialData(hostA, elemNum);
  initialData(hostB, elemNum);
  initialData(hostC, elemNum);
  memset(hostRef, 0, elemNum * sizeof(float));
  memset(gpuRef, 0, elemNum * sizeof(float));

  // 测试主机上的三个矩阵是否已经被初始化数据
  printMatrix(hostA, elemNum, nx, ny);
  printMatrix(hostB, elemNum, nx, ny);
  printMatrix(hostC, elemNum, nx, ny);

  double iStart, iElaps;

  // 在主机上执行矩阵乘法
  CGemmWithC girl;
  float alpha = 1.0;
  float beta = 1.0;
  iStart = seconds();
  girl.solveProblem(nx, nx, nx, alpha, hostA, hostB, beta, hostC, hostRef);
  iElaps = seconds();
  printf("MulMatrixOnHost Time elapsed %f sec\n", iElaps - iStart);

  // 使用cuda kernel 来执行矩阵乘法
  int blockSize = 64;
  dim3 block(blockSize);
  dim3 grid((elemNum + block.x - 1) / block.x);

  float *deviceA;
  float *deviceB;
  float *deviceC;
  CHECK(cudaMalloc((float **)&deviceA, elemNum * sizeof(float)));
  CHECK(cudaMalloc((float **)&deviceB, elemNum * sizeof(float)));
  CHECK(cudaMalloc((float **)&deviceC, elemNum * sizeof(float)));
  CHECK(cudaMemcpy(deviceA, hostA, elemNum * sizeof(float),
                   cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(deviceB, hostB, elemNum * sizeof(float),
                   cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(deviceC, hostC, elemNum * sizeof(float),
                   cudaMemcpyHostToDevice));
  iStart = seconds();
  MulMatrixOnDevice<<<block, grid>>>(nx, nx, nx, alpha, deviceA, deviceB, beta,
                                     deviceC);
  iElaps = seconds();
  printf("MulMatrixOnDevice Time elapsed %f sec\n", iElaps - iStart);
  CHECK(cudaMemcpy(gpuRef, deviceC, elemNum * sizeof(float),
                   cudaMemcpyDeviceToHost));

  checkResult(hostRef, gpuRef, elemNum);

  // 使用cublas 执行矩阵乘法
  return 0;
}
