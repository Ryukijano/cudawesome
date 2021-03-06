/*
 * @Name: vector_sum_float_opt.cu
 * @Description: Vector Floating-Point Sum
 * Custom vector dimension and block size.
 *
 * @Author: Giacomo Marciani <gmarciani@acm.org>
 * @Institution: University of Rome Tor Vergata
 *
 * @Usage: vector_sum_float_opt vectorDim blockSize
 *
 * Default values:
 *  vectorDim: 1048576
 *  blockSize: 256
 *
 * @Note: possible optimizations are:
 *  gride-stride loops: https://devblogs.nvidia.com/parallelforall/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
 */

#include <stdio.h>
#include <math.h>
#include "../../../common/error.h"
#include "../../../common/random.h"
#include "../../../common/vector.h"
#include "../../../common/mathutil.h"

#ifdef DOUBLE
#define REAL double
#else
#define REAL float
#endif

#define EPSILON (float)1e-5

__global__ void vectorAdd(const REAL *a, const REAL *b, REAL *c, const unsigned int dim) {
  const unsigned int pos = blockIdx.x * blockDim.x + threadIdx.x;

  if (pos >= dim) return;

  c[pos] = a[pos] + b[pos];
}

__host__ void gpuVectorAdd(const REAL *a, const REAL *b, REAL *c, const unsigned int vectorDim, const dim3 gridDim, const dim3 blockDim) {
  REAL *dev_a, *dev_b, *dev_c; // device copies of a, b, c
  const unsigned int size = vectorDim * sizeof(REAL); // bytes for a, b, c

  // allocate device copies of a, b, c
  HANDLE_ERROR(cudaMalloc((void**)&dev_a, size));
  HANDLE_ERROR(cudaMalloc((void**)&dev_b, size));
  HANDLE_ERROR(cudaMalloc((void**)&dev_c, size));

  // copy inputs to device
  HANDLE_ERROR(cudaMemcpyAsync(dev_a, a, size, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpyAsync(dev_b, b, size, cudaMemcpyHostToDevice));

  // launch kernel vectorAdd()
  vectorAdd<<< gridDim, blockDim >>>(dev_a, dev_b, dev_c, vectorDim);

  // copy device result back to host copy of c
  HANDLE_ERROR(cudaMemcpy(c, dev_c, size, cudaMemcpyDeviceToHost));

  // free device
  HANDLE_ERROR(cudaFree(dev_a));
  HANDLE_ERROR(cudaFree(dev_b));
  HANDLE_ERROR(cudaFree(dev_c));
}

int main(const int argc, const char **argv) {
  REAL *a, *b, *c;   // host copies of a, b, c
  unsigned int size; // bytes for a, b, c
  unsigned int vectorDim; // vector dimension
  unsigned int gridSize;  // grid size
  unsigned int blockSize; // block size
  cudaDeviceProp gpuInfo; // gpu properties

  // check arguments
  if (argc < 3) {
    fprintf(stderr, "Usage: %s vectorDim blockSize\n", argv[0]);
    exit(1);
  }

  vectorDim = atoi(argv[1]);
  blockSize = atoi(argv[2]);

  if (vectorDim < 1) {
    fprintf(stderr, "Error: vectorDim expected >= 1, got %d\n", vectorDim);
    exit(1);
  }

  if (!IS_POWER_OF_2(blockSize)) {
    fprintf(stderr, "Error: blockSize expected as power of 2, got %d\n", blockSize);
    exit(1);
  }

  // grid settings
  gridSize = (vectorDim + blockSize - 1) / blockSize;
  dim3 gridDim(gridSize);
  dim3 blockDim(blockSize);

  size = vectorDim * sizeof(REAL);

  HANDLE_ERROR(cudaGetDeviceProperties(&gpuInfo, 0));

  printf("----------------------------------\n");
  printf("Vector Floating-Point Sum\n");
  printf("----------------------------------\n");
  #ifdef DOUBLE
  printf("FP Precision: Double\n");
  #else
  printf("FP Precision: Single\n");
  #endif
  printf("Vector Dimension: %d\n", vectorDim);
  printf("Grid Size: (%d %d %d) (max: (%d %d %d))\n",
    gridDim.x, gridDim.y, gridDim.z,
    gpuInfo.maxGridSize[0], gpuInfo.maxGridSize[1], gpuInfo.maxGridSize[2]);
  printf("Block Size: (%d %d %d) (max: (%d %d %d))\n",
    blockDim.x, blockDim.y, blockDim.z,
    gpuInfo.maxThreadsDim[0], gpuInfo.maxThreadsDim[1], gpuInfo.maxThreadsDim[2]);
  printf("---------------------------------\n");

  // allocate host copies of a, b, c
  HANDLE_ERROR(cudaMallocHost((void**)&a, size));
  HANDLE_ERROR(cudaMallocHost((void**)&b, size));
  HANDLE_ERROR(cudaMallocHost((void**)&c, size));

  // fill a, b with random data
  #ifdef DOUBLE
  random_vector_double(a, vectorDim);
  random_vector_double(b, vectorDim);
  #else
  random_vector_float(a, vectorDim);
  random_vector_float(b, vectorDim);
  #endif

  // launch kernel vectorAdd()
  gpuVectorAdd(a, b, c, vectorDim, gridDim, blockDim);

  // test result
  REAL *expected;
  HANDLE_NULL(expected = (REAL*)malloc(size));
  #ifdef DOUBLE
  vector_add_double(a, b, expected, vectorDim);
  const bool correct = vector_equals_err_double(c, expected, vectorDim, EPSILON);
  #else
  vector_add_float(a, b, expected, vectorDim);
  const bool correct = vector_equals_err_float(c, expected, vectorDim, EPSILON);
  #endif
  if (!correct) {
    fprintf(stderr, "Error\n");
  } else {
    printf("Correct\n");
  }

  // free host
  HANDLE_ERROR(cudaFreeHost(a));
  HANDLE_ERROR(cudaFreeHost(b));
  HANDLE_ERROR(cudaFreeHost(c));
  free(expected);

  return 0;
}
