#include <iostream>
#include <vector>

#include "pmpp/util.hh"

// sequential merge (calling thread does all the work)
__device__ void merge_sequential(const int *A, int m, const int *B, int n, int *C) {
    int ai = 0;
    int bi = 0;
    int ci = 0;

    // neither array is exhausted
    while (ai < m && bi < n) {
        if (A[ai] <= B[bi]) {
            C[ci++] = A[ai++];
        } else {
            C[ci++] = B[bi++];
        }
    }

    // copy over remaining elements
    if (ai < m) {
        for (; ai < m; ++ai) {
            C[ci++] = A[ai];
        };
    }
    if (bi < n) {
        for (; bi < n; ++bi) {
            C[ci++] = B[bi];
        }
    }
}

/**
 * Determine the length of the leading subarray of A contributing to the leading subarray of
 * C of length k, where C is the result of merging A and B.
 * We break ties by pulling as many elements from A as possible.
 * @param k the desired rank
 * @param A sorted (ascending) array of size m
 * @param B sorted (ascending) array of size n
 * @return the length of the leading subarray of A to be used
 */
__host__ __device__ int co_rank(int k, const int *A, int m, const int *B, int n) {
    int i = std::min(k, m); // start guess in A (this is pretty arbitrary)
    int j = k - i; // start guess in B
    
    // figure out inclusive lower bound of possible indices based on lengths
    // of arrays and desired rank
    int i_lo = std::max(0, k-n);
    int j_lo = std::max(0, k-m);

    bool found = false;
    while (!found) {
        assert(0 <= i && i <= m && 0 <= j && j <= n);
        assert(i+j == k);

        if (i > 0 && j < n && A[i-1] > B[j]) {
            int delta = (i - i_lo + 1) >> 1;
            assert(delta > 0);
            j_lo = j;
            i -= delta;
            j += delta;
        } else if (i < m && j > 0 && A[i] <= B[j-1]) {
            int delta = (j - j_lo + 1) >> 1;
            assert(delta > 0);
            i_lo = i;
            i += delta;
            j -= delta;
        } else {
            found = true;
        }
    }

    return i;
}

/**
 * Merge helper. Each thread will be responsible for producing a subarray of C.
 */
__device__ void merge_by_block(const int *A, int m, const int *B, int n, int *C) {
    const int num_threads = blockDim.x;
    const int elements_per_thread = (m + n + num_threads - 1) / num_threads;

    const int C_lo = threadIdx.x * elements_per_thread;
    const int C_hi = std::min(C_lo + elements_per_thread, m + n);

    const int A_lo = co_rank(C_lo, A, m, B, n);
    const int A_hi = co_rank(C_hi, A, m, B, n); // exclusive
    const int B_lo = C_lo - A_lo;
    const int B_hi = C_hi - A_hi; // exclusive
    assert(0 <= A_lo && A_hi <= m && A_lo <= A_hi);
    assert(0 <= B_lo && B_hi <= n && B_lo <= B_hi);

    merge_sequential(A + A_lo, A_hi - A_lo, B + B_lo, B_hi - B_lo, C + C_lo);
}


__global__ void merge_tiled(const int *A, int m, const int *B, int n, int *C, int tile_size) {
    const int num_blocks = gridDim.x;
    const int num_threads = blockDim.x;

    // mildly simplifies some things
    assert(tile_size % num_threads == 0);

    // we maintain a tile of A and B in shared memory
    // these tiles roll along as we merge
    extern __shared__ int sharedAB[];
    int *A_s = &sharedAB[0];
    int *B_s = &sharedAB[tile_size];

    // each block is responsible for some chunk of the output C
    const int C_chunk_size = (m + n + num_blocks - 1) / num_blocks;
    const int C_block_lo = blockIdx.x * C_chunk_size; // start (inclusive)
    const int C_block_hi = std::min(C_block_lo + C_chunk_size, m+n); // end (exclusive)

    // determine range in A that contributes to this block's output tile
    if (threadIdx.x == 0) {
        A_s[0] = co_rank(C_block_lo, A, m, B, n);    
    } else if (threadIdx.x == 1) {
        A_s[1] = co_rank(C_block_hi, A, m, B, n);
    }
    __syncthreads();
    // these are the segments of A & B that the block needs to load and process
    const int A_block_lo = A_s[0];
    const int A_block_hi = A_s[1]; // exclusive
    const int B_block_lo = C_block_lo - A_block_lo;
    const int B_block_hi = C_block_hi - A_block_hi; // exclusive
    __syncthreads(); // need this because we reuse A_s for other purposes;
                     // could just extend shared memory by 2 and not need this

    int A_iter_lo = A_block_lo;
    int B_iter_lo = B_block_lo;

    const int num_iterations = (C_block_hi - C_block_lo + tile_size - 1) / tile_size;
    for (int iter = 0; iter < num_iterations; ++iter) {
        const int C_iter_lo = C_block_lo + tile_size * iter;
 
        // determine the actual lengths of the segments that need to be processed on this iteration
        int A_s_len = std::min(A_iter_lo + tile_size, A_block_hi) - A_iter_lo;
        int B_s_len = std::min(B_iter_lo + tile_size, B_block_hi) - B_iter_lo;
        assert(A_s_len <= tile_size && B_s_len <= tile_size);

        // load tile of A into shared memory
        for (int i=threadIdx.x; i<A_s_len; i+=num_threads) {
            int A_idx = A_iter_lo + i;
            assert(A_idx < A_block_hi);
            A_s[i] = A[A_idx];
        }

        // load tile of B into shared memory
        for (int i=threadIdx.x; i<B_s_len; i+=num_threads) {
            int B_idx = B_iter_lo + i;
            assert(B_idx < B_block_hi);
            B_s[i] = B[B_idx];
        }
        __syncthreads();

        // figure out this thread's portion of A & B
        const int merge_num_elems = tile_size / num_threads;
        int C_thread_lo = std::min(C_iter_lo + (int)threadIdx.x * merge_num_elems, C_block_hi);
        int C_thread_hi = std::min(C_thread_lo + merge_num_elems, C_block_hi);

        // figure out thread's ranks within this iteration's portion of C
        int rank_lo = C_thread_lo - C_iter_lo;
        int rank_hi = C_thread_hi - C_iter_lo;

        int i_lo = co_rank(rank_lo, A_s, A_s_len, B_s, B_s_len);
        int i_hi = co_rank(rank_hi, A_s, A_s_len, B_s, B_s_len);
        int j_lo = rank_lo - i_lo;
        int j_hi = rank_hi - i_hi;

        merge_sequential(A_s + i_lo, i_hi - i_lo, B_s + j_lo, j_hi - j_lo, C + C_thread_lo);

        __syncthreads();

        if (iter < num_iterations - 1) {
            // we could do all of this on a single thread in the block but then
            // we'd need some place in shared memory to put this
            assert(A_s_len + B_s_len >= tile_size);
            const int A_s_consumed = co_rank(tile_size, A_s, A_s_len, B_s, B_s_len);
            const int B_s_consumed = tile_size - A_s_consumed;
            A_iter_lo += A_s_consumed;
            B_iter_lo += B_s_consumed;
            __syncthreads(); // note that all/none threads in block will pass the if statement
        }
    }
}

std::vector<int> random_sorted_vector(int size) {
    std::vector<int> vec(size);
    for (int i=0; i<size; i++) {
        vec[i] = rand() % 100;
    }
    std::sort(vec.begin(), vec.end());
    return vec;
}

void test_co_rank() {
    auto A = std::vector<int>{1, 2, 3, 4};
    auto B = std::vector<int>{5,6};
    assert(0 == co_rank(0, A.data(), A.size(), B.data(), B.size()));
    assert(3 == co_rank(3, A.data(), A.size(), B.data(), B.size()));
    assert(4 == co_rank(4, A.data(), A.size(), B.data(), B.size()));
    assert(4 == co_rank(5, A.data(), A.size(), B.data(), B.size()));

    std::swap(A, B);
    assert(0 == co_rank(0, A.data(), A.size(), B.data(), B.size()));
    assert(0 == co_rank(3, A.data(), A.size(), B.data(), B.size()));
    assert(0 == co_rank(4, A.data(), A.size(), B.data(), B.size()));
    assert(1 == co_rank(5, A.data(), A.size(), B.data(), B.size()));

    A = std::vector<int>{2,4,6,8};
    B = std::vector<int>{1,3,5,7,9};
    assert(0 == co_rank(0, A.data(), A.size(), B.data(), B.size()));
    assert(0 == co_rank(1, A.data(), A.size(), B.data(), B.size()));
    assert(1 == co_rank(2, A.data(), A.size(), B.data(), B.size()));

    srand(1001);
    A = random_sorted_vector(11);
    B = random_sorted_vector(13);
    auto C = std::vector(A.begin(), A.end());
    C.insert(C.end(), B.begin(), B.end());
    std::sort(C.begin(), C.end());

    for (auto i: {0, 10, 11, 17}) { //, 70, 71, 72, 80, 84}) {
        auto ai = co_rank(i, A.data(), A.size(), B.data(), B.size());
        auto bi = i - ai;
    }
}

int main(int argc, char *argv[]) {
    test_co_rank();

    auto A = random_sorted_vector(71779);
    auto B = random_sorted_vector(91241);

    // device memory for A, B, C
    int *A_d, *B_d, *C_d;
    cudaMalloc((void**)&A_d, A.size()*sizeof(int));
    cudaMalloc((void**)&B_d, B.size()*sizeof(int));
    cudaMalloc((void**)&C_d, (A.size()+B.size())*sizeof(int));

    // copy data to device
    cudaMemcpy(A_d, A.data(), A.size()*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B.data(), B.size()*sizeof(int), cudaMemcpyHostToDevice);

    int tile_size = 1024;
    merge_tiled<<<8, 32, 2*tile_size*sizeof(int)>>>(A_d, A.size(), B_d, B.size(), C_d, tile_size);
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK_SYNC();

    // copy result back to host
    std::vector<int> C(A.size()+B.size());
    cudaMemcpy(C.data(), C_d, (A.size()+B.size())*sizeof(int), cudaMemcpyDeviceToHost);

    // check that C is sorted
    assert(std::is_sorted(C.begin(), C.end()));
}