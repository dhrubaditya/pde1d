#include <iostream>
#include <fstream>
#include <cmath>
#include <cuda_runtime.h>
#include "fft_utils.h"
#include "misc.h"
#include "random.h"
// Kernel to initialize real-space array: f(x) = sin(5x)
__global__ void init_sin_kernel(double* data, int N, double L) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        double dx = L / N;
        double x = i * dx;
        data[i] = sin(50.0 * x);
    }
}
// Kernel to set complex array
__global__ void complexify(cufftDoubleComplex* Z, double* re, double* im, int N)
{	
    int nfreqs = N / 2 + 1;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nfreqs) {
      Z[i].x = re[i];
      Z[i].y = im[i];
    }
}
// Kernel to set complex array
__global__ void set_complex(cufftDoubleComplex* Z, int N)
{	
    int nfreqs = N / 2 + 1;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nfreqs) {
      Z[i].x = 1.;
      Z[i].y = 0.;
      if (i == nfreqs -1){Z[i].y = 0.;}
      if (i == 0){Z[i].y = 0.;}
    }
}

int main(int argc, char** argv) {
    int N = 1024;              // number of real samples
    double L = 2 * M_PI;       // domain size
    if (argc > 1) {
      N = std::atoi(argv[1]);
    }
    printf("Grid size N = %d\n", N);
    // ----------------------------
    // Allocate FFT array and plan
    // ----------------------------
    FFTArray1D arr = fft_alloc_1d(N);
    // ----------------------------
    // host allocation
    // ----------------------------
    /*cufftDoubleComplex* f1;
    cufftDoubleComplex* f2;
    cudaMallocHost( (void**)&f1, sizeof(cufftDoubleComplex) * (N/2 + 1) );
    cudaMallocHost( (void**)&f2, sizeof(cufftDoubleComplex) * (N/2 + 1) );*/
    int block = 256;
    int grid = (N + block - 1) / block;
    /*init_sin_kernel<<<grid, block>>>(arr.d_real, N, L);
    GpuReducer red;
    init_reducer(red, N);
    double sum = gpu_sum(arr.d_real, N, red);
    std:: cout << sum << "\n"  ;*/
    /*double* d_real;
    double* d_imag;
    cudaMalloc(&d_real, sizeof(double) * (N/2 + 1));
    cudaMalloc(&d_imag, sizeof(double) * (N/2 + 1));
    unsigned long long seed = static_cast<unsigned long long>(time(nullptr));
    rng_init(seed);
    rng_generate_uniform(d_real, N/2 + 1);
    rng_generate_uniform(d_imag, N/2 + 1);*/
    set_complex<<<grid, block>>>(arr.d_complex, N);
    cudaDeviceSynchronize();
    GpuComplexReducer ws;
    init_Complex_reducer(ws, N);
    cufftDoubleComplex csum = gpu_Complex_sum(arr.d_complex,
                                       N, ws);
    std:: cout << csum.x << " " << csum.y << "\n"  ;
    /*cudaMemcpy(f1, arr.d_complex, 
	       sizeof(cufftDoubleComplex) * (N/2 + 1), 
	       cudaMemcpyDeviceToHost);
    cufftDoubleComplex II;
    II.x = 0;
    II.y = 1;
    complex_mult_FFTArray(arr, II); 
    // ----------------------------
    // ----------------------------
    cudaMemcpy(f2, arr.d_complex, 
		    sizeof(cufftDoubleComplex) * (N/2 + 1), 
		    cudaMemcpyDeviceToHost); */
    // ----------------------------
    // Write to file
    // ----------------------------
    /* std::ofstream fout("data.txt");
    for (int k = 0; k < N/2 + 1; ++k) {
      fout << f1[k].x << " " << f1[k].y << " " << f2[k].x << " "
	   << f2[k].y << "\n";
    }
    fout.close(); */
    // ----------------------------
    // Clean up
    // ----------------------------
    /*cudaFree(d_real);
    cudaFree(d_imag);
    cudaFreeHost(f1);
    cudaFreeHost(f2); */
    fft_free_1d(arr);
    free_Complex_reducer(ws);
    //free_reducer(red);
    std::cout << "output written to data.txt\n";
    return 0;
}
