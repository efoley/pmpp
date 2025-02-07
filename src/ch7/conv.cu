#include <cassert>
#include <iostream>
#include <vector>

#define TILE_SIZE 16

void conv2d_cpu(float *input, float *filt, float *output, int r, int width, int height) {
    for (int y=0; y<height; y++) {
        for (int x=0; x<width; x++) {
            float sum = 0.0f;
            for (int i=-r; i<=r; i++) {
                for (int j=-r; j<=r; j++) {
                    int yf = y - i;
                    int xf = x - j;
                    if (xf >= 0 && xf < width && yf >= 0 && yf < height) {
                        sum += input[yf*width+xf] * filt[(i+r)*(2*r+1) + j+r];
                    }
                }
            }
            output[y*width+x] = sum;
        }
    }
}

/**
 * 2d convolution.
 * This is implemented with tiling only (no shared memory).
 * 
 * @param input Pointer to the input data.
 * @param filt Pointer to the filter data.
 * @param output Pointer to the output data.
 * @param r radius of the filter.
 * @param width Width of the input/output.
 * @param height Height of the input/output.
 */
__global__ void conv2d_simple(float *input, float *filt, float *output, int r, int width, int height) {
    // output index
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        float sum = 0.0f;
        for (int i = -r; i <= r; ++i) {
            for (int j = -r; j <= r; ++j) {
                int yf = y - i;
                int xf = x - j;
                if (yf >= 0 && yf < height && xf >= 0 && xf < width) {
                    sum += input[yf*width + xf] * filt[(i + r)*(2*r+1) + j + r];
                }
            }
        }
        output[y * width + x] = sum;
    }
}


/**
 * 2d convolution.
 * 
 * We tile based on the output size. We load into shared memory internal cells and go to DRAM for the halo cells
 * (which hopefully will be cached.)
 */
__global__ void conv2d_output_tiling(float *input, float *filt, float *output, int r, int width, int height) {
    // output index
    auto bx = blockIdx.x, by = blockIdx.y, tx = threadIdx.x, ty = threadIdx.y;
    int x = bx * blockDim.x + tx;
    int y = by * blockDim.y + ty;

    __shared__ float in_s[TILE_SIZE][TILE_SIZE];

    if (x >= width || y >= height) {
        in_s[ty][tx] = 0.f;
    } else {
        in_s[ty][tx] = input[y * width + x];
    }
    __syncthreads();

    float sum = 0.f;

    if (x >=0 && y >=0 && x < width && y < height) {
        for (int i=-r; i<=r; ++i) {
            for (int j=-r; j<=r; ++j) {
                auto f = filt[(i+r) * (2*r+1) + (j+r)];

                int in_s_y = ty - i;
                int in_s_x = tx - j;

                float v;
                if (in_s_y >= 0 && in_s_x >= 0 && in_s_y < blockDim.y && in_s_x < blockDim.x) {
                    v = in_s[in_s_y][in_s_x];
                } else {
                    int yf = y - i;
                    int xf = x - j;

                    if (yf >= 0 && xf >= 0 && yf < height && xf < width) {
                        v = input[yf * width + xf];
                    } else {
                        v = 0.f; // Handle out-of-bounds cases appropriately
                    }
                }
                sum += f * v;
            }
        }
        output[y * width + x] = sum;
    }
}

bool approxEqual(float a, float b, float epsilon = 1e-4f) {
    return fabs(a - b) <= epsilon;
}

int main() {
    // Define the radius of the filter and dimensions of the input/output
    int r = 1;
    int width = 1000;
    int height = 800;

    std::vector<float> input(width * height);
    std::vector<float> filt0 = {1};
    std::vector<float> filt = {-0.1, -0.2, -0.1, -0.2, 1, -0.2, -0.1, -0.2, -0.1};
    assert(filt.size() == (2*r + 1) * (2*r + 1));
    std::vector<float> output_cpu(width * height, 0.f);
    std::vector<float> output_cuda(width * height, 0.f);

    // Initialize the input array with some values
    for (int i = 0; i < width * height; ++i) {
        input[i] = rand() % 256;
        //input[i] = 1.0f;
    }

    // check that r=0 copies input
    conv2d_cpu(input.data(), filt0.data(), output_cpu.data(), 0, width, height);
    for (int i = 0; i < width * height; ++i) {
        assert(output_cpu[i] == input[i]);
    }

    conv2d_cpu(input.data(), filt.data(), output_cpu.data(), r, width, height);

    // copy input to gpu
    float *input_gpu, *filt0_gpu, *filt_gpu, *output_gpu;
    cudaMalloc(&input_gpu, sizeof(float) * width * height);
    cudaMemcpy(input_gpu, input.data(), sizeof(float) * width * height, cudaMemcpyHostToDevice);
    cudaMalloc(&filt0_gpu, sizeof(float) * filt0.size());
    cudaMemcpy(filt0_gpu, filt0.data(), sizeof(float) * filt0.size(), cudaMemcpyHostToDevice);
    cudaMalloc(&filt_gpu, sizeof(float) * filt.size());
    cudaMemcpy(filt_gpu, filt.data(), sizeof(float) * filt.size(), cudaMemcpyHostToDevice);
    cudaMalloc(&output_gpu, sizeof(float) * width * height);

    dim3 threads{16, 16};
    dim3 blocks{(width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y};

    // check that r=0 copies input
    conv2d_simple<<<blocks, threads>>>(input_gpu, filt0_gpu, output_gpu, 0, width, height);
    cudaMemcpy(output_cuda.data(), output_gpu, sizeof(float) * width * height, cudaMemcpyDeviceToHost);
    for (int i = 0; i < width * height; ++i) {
        assert(output_cuda[i] == input[i]);
    }

    // check that this matches cpu
    conv2d_simple<<<blocks, threads>>>(input_gpu, filt_gpu, output_gpu, r, width, height);

    // copy output from gpu to cpu
    cudaMemcpy(output_cuda.data(), output_gpu, sizeof(float) * width * height, cudaMemcpyDeviceToHost);

    // check that results approx match
    for (int i = 0; i < width * height; ++i) {
        if (!approxEqual(output_cpu[i], output_cuda[i])) {
            std::cout << "Error: Output at index " << i << " does not match: " << output_cpu[i] << " vs. " << output_cuda[i] << std::endl;
            return -1;
        }
    }

    // check that r=0 copies input
    conv2d_output_tiling<<<blocks, threads>>>(input_gpu, filt0_gpu, output_gpu, 0, width, height);
    cudaMemcpy(output_cuda.data(), output_gpu, sizeof(float) * width * height, cudaMemcpyDeviceToHost);
    for (int i = 0; i < width * height; ++i) {
        assert(output_cuda[i] == input[i]);
    }

    // check that this matches cpu
    conv2d_output_tiling<<<blocks, threads>>>(input_gpu, filt_gpu, output_gpu, r, width, height);

    // copy output from gpu to cpu
    cudaMemcpy(output_cuda.data(), output_gpu, sizeof(float) * width * height, cudaMemcpyDeviceToHost);

    // check that results approx match
    for (int i = 0; i < width * height; ++i) {
        if (!approxEqual(output_cpu[i], output_cuda[i])) {
            std::cout << "Error 2: Output at index " << i << " does not match: " << output_cpu[i] << " vs. " << output_cuda[i] << std::endl;
            return -1;
        }
    }


}