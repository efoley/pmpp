#include <numeric>
#include <iostream>
#include <vector>

#include <cooperative_groups.h>

#include "pmpp/util.hh"

namespace cg = cooperative_groups;

/**
 * Sum elements of in for which this thread is responsible;
 * these elements are determined in the usual fashion, but we read with int4.
 */
__device__ int thread_sum(const int *in, int n) {
    int thread_id = blockIdx.x*blockDim.x + threadIdx.x;
    int global_num_threads = blockDim.x*gridDim.x;
    int4 *in4 = (int4 *) in; // TODO EDF should use bit_cast here for non-UB aliasing
    int accum = 0;
    for (int i=thread_id; i<n/4; i+=global_num_threads) {
        accum += in4[i].x + in4[i].y + in4[i].z + in4[i].w;
    }
    return accum;
}

/**
 * Sum a value over a warp.
 * This will only return the correct value on lane 0 of the warp.
 */
__device__ int warp_reduce(int v) {
    const unsigned int m = 0xffffffff; // all threads in the warp

    for (int offset=warpSize/2; offset>0; offset/=2) {
        v += __shfl_down_sync(m, v, offset);
    } 
    return v;
}

/**
 * Sums the array over each block. The output (out) array should have length
 * equal to the number of blocks and each element will contain the sum of that
 * block's input elements.
 */
__global__ void sum_by_block(const int *in, int n, int *out, int m) {
    assert(m == gridDim.x);

    // temporary workspace for writing warp sums
    assert(blockDim.x / warpSize == 32); // must be a multiple of 32 threads per block
    __shared__ int temp[32];

    int warp_idx = threadIdx.x / warpSize;

    // compute local sum
    auto s = thread_sum(in, n);

    // reduce over warp
    s = warp_reduce(s);

    if (threadIdx.x % warpSize == 0) {
        temp[warp_idx] = s; // store warp sum in shared memory
    }
    __syncthreads();

    // reduce over blocks
    if (warp_idx == 0) {
        int tx = threadIdx.x;
        assert(tx < 32);
        s = temp[tx];
        s = warp_reduce(s);

        if (tx == 0) { // store block sum in output array
            out[blockIdx.x] = s;
        }
    }

    // obviously could sum out on the device but it's simpler for now just to copy it over
}


int cuda_sum(const std::vector<int>& data, int num_blocks = 82, int num_threads_per_block = 1024) {
    // check that input size is a multiple of 4 elements, since we cast to int4 inside
    // and don't deal with danglers (obv not hard to do so)
    assert(data.size() %4 == 0);

    // allocate input array and copy data over
    int *dev_data;
    cudaMalloc(&dev_data, data.size() * sizeof(int));
    cudaMemcpy(dev_data, data.data(), data.size() * sizeof(int), cudaMemcpyHostToDevice);

    // allocate output array
    int *dev_out;
    cudaMalloc(&dev_out, num_blocks * sizeof(int));

    // launch kernel
    sum_by_block<<<num_blocks, num_threads_per_block>>>(dev_data, data.size(), dev_out, num_blocks);
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK_SYNC()

    // copy output array to host
    std::vector<int> out(num_blocks);
    cudaMemcpy(out.data(), dev_out, num_blocks * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(dev_data);
    cudaFree(dev_out);

    return std::accumulate(out.begin(), out.end(), 0);
}

void profile_cuda_sum(int data_size, int num_blocks, int num_threads_per_block, int num_warmups=2, int num_runs=100) {
    std::vector<int> data(data_size, 1);

    // sanity check kernel
    int expect_sum = std::accumulate(data.begin(), data.end(), 0);
    int sum = cuda_sum(data);
    assert(sum == expect_sum);
    std::cout << "sum: " << sum << std::endl;

    for (int i=0; i<num_warmups; i++) {
        cuda_sum(data, num_blocks, num_threads_per_block);
        CUDA_CHECK_LAST_ERROR();
        CUDA_CHECK_SYNC();
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    CUDA_CHECK_LAST_ERROR();

    CUDA_CHECK(cudaEventRecord(start));

    for (int i=0; i<num_runs; i++) {
        cuda_sum(data, num_blocks, num_threads_per_block);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    CUDA_CHECK_LAST_ERROR();

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    printf("Time: %.3f ms\n");
}

int main(int argc, char *argv[]) {
    std::vector<int> data(400000);
    srand(12414);
    // fill randomly
    for (auto &d : data) {
        d = rand()  % 100;
    }

    int expect_sum = std::accumulate(data.begin(), data.end(), 0);

    int sum = cuda_sum(data);
    assert(sum == expect_sum);

    profile_cuda_sum(400000000, 82, 1024);
}