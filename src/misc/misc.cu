#include <iostream>
#include <fstream>
#include <cmath>
#include <cuComplex.h>
#include <math.h>
#include <cuda_runtime.h>
#include "misc.h"
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error %s:%d: '%s'\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                               \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)
// ******************************************** //


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
// ----------------------------------------------------
// Initialize reducer (once)
// ----------------------------------------------------
void init_Complex_reducer(GpuComplexReducer &ws, int maxN)
{
    ws.maxN = maxN;
    ws.threads = 256;
    ws.maxBlocks = (maxN / 2 + 1 + ws.threads * 2 - 1) / (ws.threads * 2);
    cudaMalloc(&ws.d_partial, ws.maxBlocks * sizeof(cufftDoubleComplex));
}
void free_Complex_reducer(GpuComplexReducer &ws)
{
    cudaFree(ws.d_partial);
    ws.d_partial = nullptr;
}
//------------------
__global__ void reduceComplexKernel(const cufftDoubleComplex *d_in,
                                    cufftDoubleComplex *d_out,
                                    int N_half)
{
    extern __shared__ cufftDoubleComplex sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    cufftDoubleComplex sum = make_cuDoubleComplex(0.0, 0.0);

    if (i < N_half)
        sum = cuCadd(sum, d_in[i]);
    if (i + blockDim.x < N_half)
        sum = cuCadd(sum, d_in[i + blockDim.x]);

    sdata[tid] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] = cuCadd(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }

    if (tid == 0)
        d_out[blockIdx.x] = sdata[0];
}
//
cufftDoubleComplex gpu_Complex_sum(const cufftDoubleComplex *d_in,
                                       int N,
                                       GpuComplexReducer &ws)
{
    int nfreq = N / 2 + 1;
    int threads = ws.threads;
    int blocks = (nfreq + threads * 2 - 1) / (threads * 2);
    int smem = threads * sizeof(cufftDoubleComplex);

    const cufftDoubleComplex *input_ptr = d_in;
    cufftDoubleComplex result;

    // iterative reduction using preallocated buffer
    while (true) {
        reduceComplexKernel<<<blocks, threads, smem>>>(input_ptr,
						       ws.d_partial, nfreq);
        cudaDeviceSynchronize();

        if (blocks == 1) {
	  CUDA_CHECK(cudaMemcpy(&result, ws.d_partial,
				sizeof(cufftDoubleComplex),
				cudaMemcpyDeviceToHost) );
            break;
        }

        nfreq = blocks;
        blocks = (nfreq + threads * 2 - 1) / (threads * 2);
        input_ptr = ws.d_partial;
    }

    return result;
}
//----------------------
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
    extern __shared__ double sdatar[];  // shared memory for partial sums

    unsigned int tid  = threadIdx.x;
    unsigned int i    = blockIdx.x * blockDim.x + threadIdx.x;

    // Load elements into shared memory (or 0 if out of bounds)
    double x = 0.0;
    if (i < N) x = d_Arr[i];
    sdatar[tid] = x;
    __syncthreads();

    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdatar[tid] += sdatar[tid + s];
        __syncthreads();
    }

    // Write result for this block to global memory
    if (tid == 0)
        d_out[blockIdx.x] = sdatar[0];
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
//---------------------
__global__ void mult_Astar_B(cufftDoubleComplex* A,
			  cufftDoubleComplex* B, int N){
  // B = A* B
    int nfreqs = N / 2 + 1;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nfreqs) {
      cufftDoubleComplex C ;
      C.x = A[i].x * B[i].x + A[i].y * B[i].y ;
      C.y = A[i].x * B[i].y - A[i].y * B[i].x ;
      B[i].x = C.x; // stored in B
      B[i].y = C.y;
    }
}
//---------------------------------------------------------//

