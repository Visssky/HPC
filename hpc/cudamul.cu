#include <iostream>
#include <cstdlib>
#include <bits/stdc++.h>

// CUDA kernel for matrix multiplication
__global__ void matrixMultiply(int *a, int *b, int *c, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        int sum = 0;
        for (int k = 0; k < N; ++k) {
            sum += a[row * N + k] * b[k * N + col];
        }
        c[row * N + col] = sum;
    }
}

int main()
{
    int N = 4; // Matrix size

    int *a, *b, *c; // Host matrices
    int *d_a, *d_b, *d_c; // Device matrices

    int matrixSize = N * N * sizeof(int);

    // Allocate host memory
    a = (int*)malloc(matrixSize);
    b = (int*)malloc(matrixSize);
    c = (int*)malloc(matrixSize);

    // Initialize host matrices
    for (int i = 0; i < N * N; ++i) {
        a[i] = i + 1;
        b[i] = i + 1;
    }

    // Allocate device memory
    cudaMalloc((void**)&d_a, matrixSize);
    cudaMalloc((void**)&d_b, matrixSize);
    cudaMalloc((void**)&d_c, matrixSize);

    // Transfer data from host to device
    cudaMemcpy(d_a, a, matrixSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, matrixSize, cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 threadsPerBlock(2, 2);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch kernel
    matrixMultiply<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);

    // Transfer results from device to host
    cudaMemcpy(c, d_c, matrixSize, cudaMemcpyDeviceToHost);

    // Print result
    for (int i = 0; i < N * N; ++i) {
        std::cout << c[i] << " ";
        if ((i + 1) % N == 0)
            std::cout << std::endl;
    }

    // Free memory
    free(a);
    free(b);
    free(c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;}

