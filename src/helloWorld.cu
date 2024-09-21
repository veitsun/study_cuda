#include "../include/mycuda.h"
#include <cstdio>

using namespace std;

// 在CPU上调用，在GPU上执行
__global__ void helloFromGpu(void) { printf("hello world from GPU\n"); }

int main() {

  printf("hello world from CPU\n");

  helloFromGpu<<<1, 10>>>();
  cudaDeviceSynchronize(); // 确保所有内核都执行完成， 并且printf
                           // 输出被刷新到终端

  return 0;
}