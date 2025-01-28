#include <stdio.h>

__global__ void vecAddKernel(float *a, float *b, float *c, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

void vecAdd(float *a, float *b, float *c, size_t n) {
    // declare device pointers and malloc device memory for a, b, c
    float *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, n*sizeof(float));
    cudaMalloc((void**)&d_b, n*sizeof(float));
    cudaMalloc((void**)&d_c, n*sizeof(float));

    // copy data from host to device
    cudaMemcpy(d_a, a, n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n*sizeof(float), cudaMemcpyHostToDevice);

    // call kernel
    vecAddKernel<<<ceil(n/256.), 256>>>(d_a, d_b, d_c, n);

    // copy data from device to host
    cudaMemcpy(c, d_c, n*sizeof(float), cudaMemcpyDeviceToHost);

    // free device memory for a, b, c
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

int main() {
    // declare host pointers and allocate memory for a, b, c
    float a[] = {1,2,3,4};
    float b[] = {5,6,7,8};
    int n = 4;
    float c[n];
    
    vecAdd(a, b, c, n);

    // print result
    for(int i=0; i<n; i++) {
        printf("%f ", c[i]);
    }
    printf("\n");
}