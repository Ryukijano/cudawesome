/*
 * @Name: matrix_mul_nxm_int.cu
 * @Description: Matrix (NxM) Integer Product.
 * Each matrix is viewed as a single block of memory.
 * Blocks and threads are viewed as a 2D grid.
 * Custom matrix dimension and block size.
 *
 * @Author: Giacomo Marciani <gmarciani@acm.org>
 * @Institution: University of Rome Tor Vergata
 *
 * @Usage: matrix_mul_nxm_int matrixDimX1 matrixDimY1 matrixDimX2 matrixDimY2 blockSize
 *
 * Default values:
 *  matrixDimX1: 4096
 *  matrixDimY1: 4096
 *  matrixDimX2: 4096
 *  matrixDimY2: 4096
 *  blockSize: 32
 */

#include <stdio.h>
#include <math.h>
#include "../../common/error.h"
#include "../../common/random.h"
#include "../../common/matrix.h"
#include "../../common/mathutil.h"

__global__ void matrixMul(const int *a, const int *b, int *c, const unsigned int dimX1, const unsigned int dimY1, const unsigned int dimX2) {
  const unsigned int iX = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int iY = blockIdx.y * blockDim.y + threadIdx.y;

  if (iX >= dimX2 || iY >= dimY1) return;

  const unsigned int pos = iY * dimX2 + iX;

  int val = 0;
  for (unsigned int k = 0; k < dimX1; k++) {
    val += a[iY * dimX1 + k] * b[k * dimX2 + iX];
  }

  c[pos] = val;
}

__host__ void gpuMatrixMul(const int *a, const int *b, int *c,
    const unsigned int matrixDimX1, const unsigned int matrixDimY1, const unsigned int matrixDimX2, const unsigned int matrixDimY2,
    const dim3 gridDim, const dim3 blockDim) {
  int *dev_a, *dev_b, *dev_c; // device copies of a, b, c
  const unsigned int size_a = matrixDimX1 * matrixDimY1 * sizeof(int); // bytes for a
  const unsigned int size_b = matrixDimX2 * matrixDimY2 * sizeof(int); // bytes for b
  const unsigned int size_c = matrixDimY1 * matrixDimX2 * sizeof(int); // bytes for c

  // allocate device copy of a, b, c
  HANDLE_ERROR(cudaMalloc((void**)&dev_a, size_a));
  HANDLE_ERROR(cudaMalloc((void**)&dev_b, size_b));
  HANDLE_ERROR(cudaMalloc((void**)&dev_c, size_c));

  // copy inputs to device
  HANDLE_ERROR(cudaMemcpy(dev_a, a, size_a, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dev_b, b, size_b, cudaMemcpyHostToDevice));

  // launch mul() kernel
  matrixMul<<< gridDim, blockDim >>>(dev_a, dev_b, dev_c, matrixDimX1, matrixDimY1, matrixDimX2);

  // copy device result back to host copy of c
  HANDLE_ERROR(cudaMemcpy(c, dev_c, size_c, cudaMemcpyDeviceToHost));

  // free device
  HANDLE_ERROR(cudaFree(dev_a));
  HANDLE_ERROR(cudaFree(dev_b));
  HANDLE_ERROR(cudaFree(dev_c));
}

int main(const int argc, const char **argv) {
  int *a, *b, *c;         // host copies of a, b, c
  unsigned int size_a, size_b, size_c; // bytes for a, b, c
  unsigned int matrixDimX1, matrixDimY1, matrixDimX2, matrixDimY2; // matrices dimensions
  unsigned int gridSizeX, gridSizeY; // grid size
  unsigned int blockSize; // block size
  cudaDeviceProp gpuInfo; // gpu properties

  // check arguments
  if (argc < 6) {
    fprintf(stderr, "Usage: %s matrixDimX1 matrixDimY1 matrixDimX2 matrixDimY2 blockSize\n", argv[0]);
    exit(1);
  }

  matrixDimX1 = atoi(argv[1]);
  matrixDimY1 = atoi(argv[2]);
  matrixDimX2 = atoi(argv[3]);
  matrixDimY2 = atoi(argv[4]);
  blockSize = atoi(argv[5]);

  if (matrixDimX1 < 1) {
    fprintf(stderr, "Error: matrixDimX1 expected >= 1, got %d\n", matrixDimX1);
    exit(1);
  }

  if (matrixDimY1 < 1) {
    fprintf(stderr, "Error: matrixDimY1 expected >= 1, got %d\n", matrixDimY1);
    exit(1);
  }

  if (matrixDimX2 < 1) {
    fprintf(stderr, "Error: matrixDimX2 expected >= 1, got %d\n", matrixDimX2);
    exit(1);
  }

  if (matrixDimY2 != matrixDimX1) {
    fprintf(stderr, "Error: matrixDimY2 expected = matrixDimX1 (%d), got %d\n", matrixDimX1, matrixDimY2);
    exit(1);
  }

  if (!IS_POWER_OF_2(blockSize)) {
    fprintf(stderr, "Error: blockSize expected as power of 2, got %d\n", blockSize);
    exit(1);
  }

  // grid settings
  const unsigned int maxDimX = max(matrixDimX1, matrixDimX2);
  gridSizeX = maxDimX / blockSize;
  if (gridSizeX * blockSize < maxDimX) {
     gridSizeX += 1;
  }
  const unsigned int maxDimY = max(matrixDimY1, matrixDimY2);
  gridSizeY = maxDimY / blockSize;
  if (gridSizeY * blockSize < maxDimY) {
     gridSizeY += 1;
  }
  dim3 gridDim(gridSizeX, gridSizeY);
  dim3 blockDim(blockSize, blockSize);

  size_a = matrixDimX1 * matrixDimY1 * sizeof(int);
  size_b = matrixDimX2 * matrixDimY2 * sizeof(int);
  size_c = matrixDimY1 * matrixDimX2 * sizeof(int);

  HANDLE_ERROR(cudaGetDeviceProperties(&gpuInfo, 0));

  printf("------------------------------------\n");
  printf("Matrix (NxM) Integer Product\n");
  printf("------------------------------------\n");
  printf("Matrix Dimension (A): (%d, %d)\n", matrixDimX1, matrixDimY1);
  printf("Matrix Dimension (B): (%d, %d)\n", matrixDimX2, matrixDimY2);
  printf("Grid Size: (%d, %d, %d) (max: (%d, %d, %d))\n",
    gridDim.x, gridDim.y, gridDim.z,
    gpuInfo.maxGridSize[0], gpuInfo.maxGridSize[1], gpuInfo.maxGridSize[2]);
  printf("Block Size: (%d, %d, %d) (max: (%d, %d, %d))\n",
    blockDim.x, blockDim.y, blockDim.z,
    gpuInfo.maxThreadsDim[0], gpuInfo.maxThreadsDim[1], gpuInfo.maxThreadsDim[2]);
  printf("-----------------------------------\n");

  // allocate host copy of a, b, c
  HANDLE_NULL(a = (int*)malloc(size_a));
  HANDLE_NULL(b = (int*)malloc(size_b));
  HANDLE_NULL(c = (int*)malloc(size_c));

  // fill a, b with random data
  random_matrix_int(a, matrixDimX1, matrixDimY1);
  random_matrix_int(b, matrixDimX2, matrixDimY2);

  // launch kernel matrixMul()
  gpuMatrixMul(a, b, c, matrixDimX1, matrixDimY1, matrixDimX2, matrixDimY2, gridDim, blockDim);

  // test result
  int *expected;
  HANDLE_NULL(expected = (int*)malloc(size_c));
  matrix_mul_int(a, b, expected, matrixDimX1, matrixDimY1, matrixDimX2);
  const bool correct = matrix_equals_int(c, expected, matrixDimX2, matrixDimY1);
  if (!correct) {
    fprintf(stderr, "Error\n");
  } else {
    printf("Correct\n");
  }

  // free host
  free(a);
  free(b);
  free(c);
  free(expected);

  return 0;
}
