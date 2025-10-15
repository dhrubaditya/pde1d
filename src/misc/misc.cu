#include <iostream>
#include <fstream>
#include <cmath>
#include <cuComplex.h>
#include <math.h>
#include <cuda_runtime.h>
#include "misc.h"


__device__ double gaussian(double x, double x_0, double sigma, double A) 
{
    // Gaussian centered at x_0, width parameter sigma, amplitude A
    double arg = (x - x_0) / (sigma);
    return A * exp(-arg * arg);
}
//
__device__ cufftDoubleComplex cuCpow(cufftDoubleComplex z, double n) {
    double r = cuCabs(z);          // magnitude = sqrt(x^2 + y^2)
    double theta = atan2(z.y, z.x); // argument = phase
    double rn = pow(r, n);         // magnitude^n
    double ang = n * theta;        // n * phase

    return make_cuDoubleComplex(rn * cos(ang), rn * sin(ang));
}

//------------------------------------------------//
void clean_exit_host(const std::string &msg,
			      bool mark_incomplete){
  
  if (msg.empty()){
        std::cerr << "Exiting" << std::endl;
  }else{
        std::cerr << "Exiting: " << msg << std::endl;
  }
  if (mark_incomplete) {
    std::ofstream marker("INCOMPLETE");
    if (marker.is_open()) {
      marker << "Program exited early via clean_exit_host()." << std::endl;
      if (!msg.empty())
	marker << "Reason: " << msg << std::endl;
      marker.close();
      std::cerr << "Created marker file: INCOMPLETE" << std::endl;
    } else {
      std::cerr << "Warning: failed to create INCOMPLETE file." << std::endl;
    }
  }
    
  // Ensure all device resources are released
  cudaDeviceReset();
  
  // Exit cleanly
  exit(EXIT_SUCCESS);
}
//
void init_reducer(GpuReducer &red, size_t N) {
    const int BLOCK_SIZE = 256;
    size_t gridSize = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    red.max_size = gridSize;
    cudaMalloc(&red.d_partial, gridSize * sizeof(double));
    red.h_partial = (double*)malloc(gridSize * sizeof(double));
}
//
void free_reducer(GpuReducer &red) {
    cudaFree(red.d_partial);
    free(red.h_partial);
    red.d_partial = nullptr;
    red.h_partial = nullptr;
}
//
__global__ void reduce_sum(const double *d_Arr, double *d_out, size_t N) {
    extern __shared__ double sdata[];  // shared memory for partial sums

    unsigned int tid  = threadIdx.x;
    unsigned int i    = blockIdx.x * blockDim.x + threadIdx.x;

    // Load elements into shared memory (or 0 if out of bounds)
    double x = 0.0;
    if (i < N) x = d_Arr[i];
    sdata[tid] = x;
    __syncthreads();

    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // Write result for this block to global memory
    if (tid == 0)
        d_out[blockIdx.x] = sdata[0];
}

//
double gpu_sum(const double* d_Arr, size_t N, GpuReducer &red) {
    const int BLOCK_SIZE = 256;
    int gridSize = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    reduce_sum<<<gridSize, BLOCK_SIZE, BLOCK_SIZE * sizeof(double)>>>(d_Arr, red.d_partial, N);

    size_t n = gridSize;
    while (n > 1) {
        int newGrid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        reduce_sum<<<newGrid, BLOCK_SIZE, BLOCK_SIZE * sizeof(double)>>>(red.d_partial, red.d_partial, n);
        n = newGrid;
    }

    cudaMemcpy(red.h_partial, red.d_partial, sizeof(double), cudaMemcpyDeviceToHost);
    return red.h_partial[0];
}
//
__host__ __device__  cufftDoubleComplex exp_cuComplex(cufftDoubleComplex G,
						      double t)
{
    double a = G.x; // real part
    double b = G.y; // imag part
    double exp_real = exp(a * t);
    cufftDoubleComplex result;
    result.x = exp_real * cos(b * t);
    result.y = exp_real * sin(b * t);
    return result;
}


