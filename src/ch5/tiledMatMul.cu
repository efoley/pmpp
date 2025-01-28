#include <stdio.h>

#define TILE_SIZE 16


__global__ void tiledMatMulKernel(float *A, float *B, float *C, size_t M, size_t N, size_t P) {
    auto by = blockIdx.y, bx = blockIdx.x;
    auto ty = threadIdx.y, tx = threadIdx.x;

    __shared__ float A_block[TILE_SIZE][TILE_SIZE];
    __shared__ float B_block[TILE_SIZE][TILE_SIZE];

    // starting row/col for this block
    auto block_row = by * TILE_SIZE;
    auto block_col = bx * TILE_SIZE;

    // row/col of output for this thread 
    auto row = block_row + ty;
    auto col = block_col + tx;

    float sum = 0.0f;
    for (size_t off = 0; off < N; off += TILE_SIZE) {
        // load A and B into shared memory
        // (or zero if out of bounds)
        if (row < M && off + tx < N) {
            A_block[ty][tx] = A[row * N + off + tx];
        } else {
            A_block[ty][tx] = 0.0f;
        }
        if (off + ty < N && col < P) {
            B_block[ty][tx] = B[(off + ty) * P + col];
        } else {
            B_block[ty][tx] = 0.0f;
        }
        __syncthreads();

        // perform matrix multiplication in shared memory
        for (int i=0; i<TILE_SIZE; i++) {
            sum += A_block[ty][i] * B_block[i][tx];
        }
        __syncthreads();
    }

    // write result to global memory
    if (row < M && col < P) {
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
    dim3 threads(TILE_SIZE, TILE_SIZE, 1);
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
    tiledMatMulKernel<<<blocks, threads>>>(A_d, B_d, C_d, M, N, P);


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