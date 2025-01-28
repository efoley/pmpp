#include <stdio.h>

/**
 * MatMulKernel: Performs matrix multiplication on the GPU.
 * 
 * Should run with one thread per element of the output matrix.
 *
 * @param A: Pointer to the first matrix.
 * @param B: Pointer to the second matrix.
 * @param C: Pointer to the result matrix.
 * @param M: Number of rows in A and C.
 * @param N: Number of columns in A.
 * @param P: Number of columns in B and C.
 */
__global__ void matMulKernel(float *A, float *B, float *C, size_t M, size_t N, size_t P) {
    // output index
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    // if the output index is within bounds of the matrices
    if (row < M && col < P) {
        float sum = 0.0f;
        for (size_t k = 0; k < N; ++k) {
            // NOTE that the access of B is not done efficiently here;
            // it would be better to take as input B transposed. 
            sum += A[row * N + k] * B[k * P + col];
        }
        C[row * P + col] = sum;
    }
}

/**
 * Multiply two matrices.
 *
 * @param A: Pointer to the first matrix.
 * @param B: Pointer to the second matrix.
 * @param C: Pointer to the result matrix.
 * @param M: Number of rows in A and C.
 * @param N: Number of columns in A.
 * @param P: Number of columns in B and C.
 */
void matMul(float *A, float *B, float *C, size_t M, size_t N, size_t P) {
    // declare grid and block dimensions for the CUDA kernel
    dim3 threads(16, 16, 1);
    dim3 blocks((P + threads.y - 1) / threads.y,
                (M + threads.x - 1) / threads.x, 1); 

    // setup device arrays
    float *A_d, *B_d, *C_d;
    cudaMalloc((void**)&A_d, M*N*sizeof(float));
    cudaMalloc((void**)&B_d, N*P*sizeof(float));
    cudaMalloc((void**)&C_d, M*P*sizeof(float));

    // copy data from host to device
    cudaMemcpy(A_d, A, M*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, N*P*sizeof(float), cudaMemcpyHostToDevice);

    // call kernel
    matMulKernel<<<blocks, threads>>>(A_d, B_d, C_d, M, N, P);


    // copy data from device to host
    cudaMemcpy(C, C_d, M*P*sizeof(float), cudaMemcpyDeviceToHost);

    // free device memory 
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

int main() {
    // make a 5x3 matrix A and a 3x2 matrix B
    float A[] = {1, 2,3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5};
    float B[] = {1, 2, 2, 4, 4, 8};
    // initialzie C to 10 zeros
    float C[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    int M = 5;
    int N = 3;
    int P = 2;

    // call matrix multiplication function
    matMul(A, B, C, M, N, P);

    // print out C
    for (int i=0; i<M*P; i++) {
        printf("%f ", C[i]);
    }
    printf("\n");
}