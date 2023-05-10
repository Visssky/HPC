#include <iostream>
#include <cuda_runtime.h>
#include <bits/stdc++.h>
// Kernel function for vector addition
__global__ void vectorAdd(const float* a, const float* b, float* c, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        c[idx] = a[idx] + b[idx];
}

int main()
{
    int size = 1000000;  // Size of the vectors
    size_t bytes = size * sizeof(float);

    // Allocate memory on the host (CPU)
    float* h_a = new float[size];
    float* h_b = new float[size];
    float* h_c = new float[size];

    // Initialize input vectors
    for (int i = 0; i < size; ++i) {
        h_a[i] = i;
        h_b[i] = i;
    }

    // Allocate memory on the device (GPU)
    float* d_a, * d_b, * d_c;
    cudaMalloc((void**)&d_a, bytes);
    cudaMalloc((void**)&d_b, bytes);
    cudaMalloc((void**)&d_c, bytes);

    // Copy input data from host to device
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // Define block and grid sizes
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel on the GPU
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, size);

    // Copy result from device to host
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // Print the first 10 elements of the result
    for (int i = 0; i < 10; ++i) {
        std::cout << h_c[i] << " ";
    }
    std::cout << std::endl;

    // Free memory
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
