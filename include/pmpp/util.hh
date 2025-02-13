#include <cassert>
#include <iostream>

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
#define CUDA_CHECK_LAST_ERROR() { CUDA_CHECK(cudaPeekAtLastError()); }

// useful to check for kernel run errors
#define CUDA_CHECK_SYNC() { CUDA_CHECK(cudaDeviceSynchronize()); }