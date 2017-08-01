/*
 * @Name: addvec_threads.cu
 * @Description: Integer vectors addition with CUDA.
 * One block, multiple threads per block.
 *
 * @Author: Giacomo Marciani <gmarciani@acm.org>
 */

#include <stdio.h>
#include <math.h>
#include <../common/error.h>

#define N 512

__global__ void add(int *a, int *b, int *c) {
  c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
}

void random_ints(int *p, int n) {
  int i;
  for(i = 0; i<n; i++) {
    p[i] = rand();
  }
}

int main( void ) {
  int *a, *b, *c;         // host copies of a, b, c
  int *dev_a, *dev_b, *dev_c; // device copies of a, b, c
  int size = N * sizeof(int); // size of N integers
  int i;

  // allocate device copies of a, b, c
  HANDLE_ERROR(cudaMalloc((void**)&dev_a, size));
  HANDLE_ERROR(cudaMalloc((void**)&dev_b, size));
  HANDLE_ERROR(cudaMalloc((void**)&dev_c, size));

  a = HANDLE_NULL((int*)malloc(size));
  b = HANDLE_NULL((int*)malloc(size));
  c = HANDLE_NULL((int*)malloc(size));

  random_ints(a, N);
  random_ints(b, N);

  // copy inputs to device
  HANDLE_ERROR(cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice));

  // launch add() kernel with N parallel blocks
  add<<< 1, N >>>(dev_a, dev_b, dev_c);

  // copy device result back to host copy of c
  HANDLE_ERROR(cudaMemcpy(c, dev_c, size, cudaMemcpyDeviceToHost));

  for(i = 0; i < N; i++) {
    if(c[i] != a[i] + b[i]) {
      printf("error: expected %d, got %d!\n",c[i], d[i]);
      break;
    }
  }

  free(a);
  free(b);
  free(c);

  HANDLE_ERROR(cudaFree(dev_a));
  HANDLE_ERROR(cudaFree(dev_b));
  HANDLE_ERROR(cudaFree(dev_c));

  return 0;
}
