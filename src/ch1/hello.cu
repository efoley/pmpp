#include <stdio.h>

__global__ void hello() {
    printf("Hello from CUDA!\n");
}

int main() {
    // Launch the kernel on a single thread
    hello<<<1, 1>>>();
    cudaDeviceSynchronize(); // Wait for the kernel to finish
    return 0;
}