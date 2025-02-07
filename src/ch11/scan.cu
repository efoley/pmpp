#include <cassert>
#include <iostream>
#include <vector>

#define CUDA_CHECK(ans) { cuda_assert((ans), __FILE__, __LINE__); }
inline void cuda_assert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// useful to check for kernel launch errors
#define CUDA_CHECK_LAUNCH() { CUDA_CHECK(cudaPeekAtLastError()); }

// useful to check for kernel run errors
#define CUDA_CHECK_SYNC() { CUDA_CHECK(cudaDeviceSynchronize()); }

const int SECTION_SIZE = 1024;

/**
 * Performs an inclusive scan on a section of the input array, where
 * SECTION_SIZE determines how many elements are processed in each iteration. The
 * result is stored in the output array. This function assumes that the input and
 * output arrays ve the same size.
 *
 * @param in The input array.
 * @param out The output array where the result will be stored.
 * @param n The number of elements to scan.
 */
__global__ void kogge_section_inclusive_scan(const float *in, float *out, int n)
{
    assert(blockDim.x == SECTION_SIZE);

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int t = threadIdx.x;
    __shared__ float work[SECTION_SIZE];

    if (i < n) {
        work[t] = in[i];
    } else {
        work[t] = 0.0f;
    }

    for (int stride=1; stride < SECTION_SIZE; stride *= 2) {
        __syncthreads();
        float temp = work[t];
        if (t >= stride) {
            temp += work[t-stride];
        }
        __syncthreads();
        work[t] = temp;
    }

    if (i < n) {
        out[i] = work[t];
    }
}

const int COARSE_BLOCK_SIZE = 256;
const int COARSE_FACTOR = 4;
const int COARSE_SECTION_SIZE = COARSE_BLOCK_SIZE * COARSE_FACTOR;

__global__ void kogge_section_inclusive_scan_coarse(const float *in, float *out, int n) {
    assert(blockDim.x == COARSE_BLOCK_SIZE);


    __shared__ float work[COARSE_SECTION_SIZE];
    const int nt = blockDim.x;
    const int t = threadIdx.x;

    const int in_lo = blockIdx.x * COARSE_SECTION_SIZE;

    // load data into shared memory
    // do this with coalesced reads
    for (int i = t; i < COARSE_SECTION_SIZE; i+= nt) {
        if (in_lo + i < n) {
            work[i] = in[in_lo + i];
        } else {
            work[i] = 0.0f;
        }
    }

    __syncthreads();

    // do serial thread-local scan on COARSE_FACTOR-length segments
    const int work_lo = t * COARSE_FACTOR;
    for (int i=work_lo + 1; i<work_lo + COARSE_FACTOR; i++) {
        work[i] += work[i-1];
    }

    __syncthreads();

    // do kogge-stone scan, but only on the last element of each section
    for (int stride=1; stride < COARSE_BLOCK_SIZE; stride*=2) {
        __syncthreads();
        int i = (t + 1) * COARSE_FACTOR - 1; 
        float temp = work[i];
        if (t >= stride) {
            assert(i - stride * COARSE_FACTOR >= 0);
            temp += work[i - stride * COARSE_FACTOR];
        }
        __syncthreads();
        work[i] = temp;
    }

    __syncthreads();

    // fix elements of all but the first section
    // note that we skip the last element of each section since
    // it has the correct value from the coarse scan above
    if (work_lo > 0) {
        float a = work[work_lo - 1];
        for (int i=work_lo; i<work_lo + COARSE_FACTOR-1; i++) {
            work[i] += a;
        }
    }

    __syncthreads();

    // write section to output
    // do this with coalesced writes
    for (int i=t; i<COARSE_SECTION_SIZE; i+=nt) {
        if (in_lo + i < n) {
            out[in_lo + i] = work[i];
        }
    }
}

int main(int argc, char *argv[]) {
    std::vector<float> input(2000, 1.0f); 
    std::vector<float> output(input.size());

    // create device arrays
    float *d_input, *d_output;
    cudaMalloc((void**)&d_input, input.size() * sizeof(float));
    cudaMalloc((void**)&d_output, output.size() * sizeof(float));

    CUDA_CHECK(cudaMemcpy(d_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice));

    {
        dim3 blocks((input.size() + SECTION_SIZE - 1) / SECTION_SIZE);
        kogge_section_inclusive_scan<<<blocks, SECTION_SIZE>>>(d_input, d_output, input.size());
    }
    CUDA_CHECK(cudaMemcpy(output.data(), d_output, output.size() * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i=0; i<output.size(); ++i) {
        if (output[i] != (i%SECTION_SIZE) + 1) {
            std::cout << "Error 1: Output[" << i << "] = " << output[i] << ", expected " << i + 1 << std::endl;
        }
    }

    CUDA_CHECK(cudaMemset(d_output, 0, output.size() * sizeof(float)));

    {
        dim3 blocks((input.size() + COARSE_SECTION_SIZE - 1) / COARSE_SECTION_SIZE);
        kogge_section_inclusive_scan_coarse<<<blocks, COARSE_BLOCK_SIZE>>>(d_input, d_output, input.size());
        CUDA_CHECK_LAUNCH();
        CUDA_CHECK_SYNC();
    }

    CUDA_CHECK(cudaMemcpy(output.data(), d_output, output.size() * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i=0; i<output.size(); ++i) {
        if (output[i] != (i%SECTION_SIZE) + 1) {
            std::cout << "Error 2: Output[" << i << "] = " << output[i] << ", expected " << i + 1 << std::endl;
        }
    }
}