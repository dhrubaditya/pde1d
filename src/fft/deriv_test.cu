#include "fft_utils.h"

#include <iostream>
#include <fstream>
#include <cmath>
#include <cuda_runtime.h>

// Kernel to initialize real-space array: f(x) = sin(5x)
__global__ void init_sin_kernel(double* data, int N, double L) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        double dx = L / N;
        double x = i * dx;
        data[i] = cos(2*x);
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
    init_sin_kernel<<<grid, block>>>(arr.d_real, N, L);
    cudaDeviceSynchronize();
    // copy to host
    double* f = new double[N + 2];
    cudaMemcpy(f, arr.d_real, sizeof(double) * (N + 2), cudaMemcpyDeviceToHost);
    
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
    double* df = new double[N + 2];
    cudaMemcpy(df, arr.d_real, sizeof(double) * (N + 2), cudaMemcpyDeviceToHost);

    // ----------------------------
    // Write to file
    // ----------------------------
    std::ofstream fout("data.txt");
    for (int k = 0; k < N; ++k) {
      fout << k << " " << f[k] << " " << df[k] << "\n";
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
