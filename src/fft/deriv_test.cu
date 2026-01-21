#include "fft_utils.h"

#include <iostream>
#include <fstream>
#include <cmath>
#include <cuda_runtime.h>

// Kernel to initialize real-space array: f(x) = sin(5x)
__global__ void init_eix_kernel(cufftDoubleComplex* data, int N, double L) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        double dx = L / N;
        double x = i * dx;
	int kf = 5.;
        data[i].x = cos(kf * x);
	data[i].y = sin(kf * x);
    }
}
int main(int argc, char** argv){
    int N = 128;              // number of real samples
    double L = 2 * M_PI;       // domain size
    if (argc > 1) {
      N = std::atoi(argv[1]);
      if (N <= 0) {
        printf("Error: N must be a positive integer.\n");
        return 1;
       }
     }
     printf("Grid size N = %d\n", N);
     //test_fft_freq(N);
     
    // ----------------------------
    // Allocate FFT array and plan
    // ----------------------------
    FFTArray1D arr = fft_alloc_1d(N);
    FFTPlan1D plan = fft_plan_create_1d(N);

    // ----------------------------
    // Initialize array directly on device in real space
    // ----------------------------
    arr.IsFourier = false;
    int block = 256;
    int grid = (N + block - 1) / block;
    init_eix_kernel<<<grid, block>>>(arr.d_complex, N, L);
    cudaDeviceSynchronize();
    // copy to host
    cufftDoubleComplex* f = new cufftDoubleComplex[N ];
    cudaMemcpy(f, arr.d_complex, sizeof(cufftDoubleComplex) * N ,
	            cudaMemcpyDeviceToHost);
    
    // ----------------------------
    // Forward FFT (real -> complex)
    // ----------------------------
    fft_forward_inplace(plan, arr);
    // ----------------------------
    // Calculate its first derivative and store in-place
    // ----------------------------
    derivk(arr, -1./4., true);
    // then inverse fft
    fft_inverse_inplace(plan, arr);
    normalize_fft(arr);
    // copy back to host
    cufftDoubleComplex* df = new cufftDoubleComplex[N ];
    cudaMemcpy(df, arr.d_complex, sizeof(cufftDoubleComplex) * N ,
	       cudaMemcpyDeviceToHost);

    // ----------------------------
    // Write to file
    // ----------------------------
    std::ofstream fout("data.txt");
    for (int k = 0; k < N; ++k) {
      fout << k << " " << f[k].x << " " <<f[k].y << " "
	   << df[k].x << " " << df[k].y << "\n";
    }
    fout.close();
    delete[] f;
    delete[] df;

    // ----------------------------
    // Clean up
    // ----------------------------
    fft_free_1d(arr);
    fft_plan_destroy_1d(plan);

    std::cout << "func and its deriv in data.txt\n"; 
    return 0;
}
