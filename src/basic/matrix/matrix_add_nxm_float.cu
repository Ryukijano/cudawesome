/*
 * @Name: matrix_add_nxm_float.cu
 * @Description: Matrix (NxM) Floating-Point Sum.
 * Each matrix is viewed as a single block of memory.
 * Blocks and threads are viewed as a 2D grid.
 * Custom matrix dimensions and block size.
 *
 * @Author: Giacomo Marciani <gmarciani@acm.org>
 * @Institution: University of Rome Tor Vergata
 *
 * @Usage: matrix_add_nxm_float matrixDimX matrixDimY blockSize
 *
 * Default values:
 *  matrixDimX: 4096
 *  matrixDimY: 4096
 *  blockSize: 32
 */

#include <stdio.h>
#include <math.h>
#include "../../common/error.h"
#include "../../common/random.h"
#include "../../common/matrix.h"

#ifdef DOUBLE
#define REAL double
#else
#define REAL float
#endif

__global__ void add(const REAL *a, const REAL *b, REAL *c, const unsigned int dimX, const unsigned int dimY) {
  const unsigned int iX = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int iY = blockIdx.y * blockDim.y + threadIdx.y;

  if (iX < dimX && iY < dimY) {
    const unsigned int pos = iY * dimX + iX;
    c[pos] = a[pos] + b[pos];
  }
}

int main(const int argc, const char **argv) {
  REAL *a, *b, *c;             // host copies of a, b, c
  REAL *dev_a, *dev_b, *dev_c; // device copies of a, b, c
  unsigned int size; // bytes for a, b, c
  unsigned int matrixDimX, matrixDimY; // matrix dimensions
  unsigned int gridSizeX, gridSizeY; // grid size
  unsigned int blockSize; // block size
  cudaDeviceProp gpuInfo; // gpu properties

  if (argc < 4) {
    fprintf(stderr, "Usage: %s matrixDimX matrixDimY blockSize\n", argv[0]);
    exit(1);
  }

  matrixDimX = atoi(argv[1]);
  matrixDimY = atoi(argv[2]);
  blockSize = atoi(argv[3]);

  if (matrixDimX < 1) {
    fprintf(stderr, "Error: matrixDimX expected >= 1, got %d\n", matrixDimX);
    exit(1);
  }

  if (matrixDimY < 1) {
    fprintf(stderr, "Error: matrixDimY expected >= 1, got %d\n", matrixDimY);
    exit(1);
  }

  if (blockSize < 1) {
    fprintf(stderr, "Error: blockSize expected >= 1, got %d\n", blockSize);
    exit(1);
  }

  // grid settings
  gridSizeX = matrixDimX / blockSize;
  if (gridSizeX * blockSize < matrixDimX) {
     gridSizeX += 1;
  }
  gridSizeY = matrixDimY / blockSize;
  if (gridSizeY * blockSize < matrixDimY) {
     gridSizeY += 1;
  }
  dim3 gridDim(gridSizeX, gridSizeY);
  dim3 blockDim(blockSize, blockSize);

  size = matrixDimX * matrixDimY * sizeof(REAL);

  HANDLE_ERROR(cudaGetDeviceProperties(&gpuInfo, 0));

  printf("------------------------------------\n");
  printf("Matrix (NxM) Floating-Point Sum\n");
  printf("------------------------------------\n");
  #ifdef DOUBLE
  printf("FP Precision: Double\n");
  #else
  printf("FP Precision: Single\n");
  #endif
  printf("Matrix Dimension: (%d, %d)\n", matrixDimX, matrixDimY);
  printf("Grid Size: (%d, %d, %d) (max: (%d, %d, %d))\n",
    gridDim.x, gridDim.y, gridDim.z,
    gpuInfo.maxGridSize[0], gpuInfo.maxGridSize[1], gpuInfo.maxGridSize[2]);
  printf("Block Size: (%d, %d, %d) (max: (%d, %d, %d))\n",
    blockDim.x, blockDim.y, blockDim.z,
    gpuInfo.maxThreadsDim[0], gpuInfo.maxThreadsDim[1], gpuInfo.maxThreadsDim[2]);
  printf("-----------------------------------\n");

  // allocate host copies of a, b, c
  HANDLE_NULL(a = (REAL*)malloc(size));
  HANDLE_NULL(b = (REAL*)malloc(size));
  HANDLE_NULL(c = (REAL*)malloc(size));

  // allocate device copies of a, b, c
  HANDLE_ERROR(cudaMalloc((void**)&dev_a, size));
  HANDLE_ERROR(cudaMalloc((void**)&dev_b, size));
  HANDLE_ERROR(cudaMalloc((void**)&dev_c, size));

  // fill a, b with random data
  #ifdef DOUBLE
  random_matrix_double(a, matrixDimX, matrixDimY);
  random_matrix_double(b, matrixDimX, matrixDimY);
  #else
  random_matrix_float(a, matrixDimX, matrixDimY);
  random_matrix_float(b, matrixDimX, matrixDimY);
  #endif

  // copy inputs to device
  HANDLE_ERROR(cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice));

  // launch add() kernel
  add<<< gridDim, blockDim >>>(dev_a, dev_b, dev_c, matrixDimX, matrixDimY);

  // copy device result back to host copy of c
  HANDLE_ERROR(cudaMemcpy(c, dev_c, size, cudaMemcpyDeviceToHost));

  // test result
  REAL *expected;
  HANDLE_NULL(expected = (REAL*)malloc(size));
  #ifdef DOUBLE
  matrix_add_double(a, b, expected, matrixDimX, matrixDimY);
  const bool equal = matrix_equals_double(c, expected, matrixDimX, matrixDimY);
  #else
  matrix_add_float(a, b, expected, matrixDimX, matrixDimY);
  const bool equal = matrix_equals_float(c, expected, matrixDimX, matrixDimY);
  #endif
  if (!equal) {
    fprintf(stderr, "Error\n");
  } else {
    printf("Correct\n");
  }

  // free host
  free(a);
  free(b);
  free(c);
  free(expected);

  // free device
  HANDLE_ERROR(cudaFree(dev_a));
  HANDLE_ERROR(cudaFree(dev_b));
  HANDLE_ERROR(cudaFree(dev_c));

  return 0;
}