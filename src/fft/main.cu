#include "fft_utils.h"

#include <iostream>
#include <fstream>
#include <cmath>
#include <cuda_runtime.h>

// Kernel to initialize real-space array: f(x) = sin(5x)
__global__ void init_sin_kernel(cufftDoubleComplex* data,
				int N, double L, int kf) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        double dx = L / N;
        double x = i * dx;
        data[i].x = sin((double) kf * x);
        data[i].y = 0.;
    }
}

int main(int argc, char* argv[]){
    int N = 128;              // number of real samples
    int kf = 4;
    double L = 2 * M_PI;       // domain size
    N = std::atoi(argv[1]);
    kf = std::atoi(argv[2]);
    printf("Grid size N = %d\n", N);
    printf("kf  = %d\n", kf);
    // ----------------------------
    // Allocate FFT array and plan
    // ----------------------------
    FFTArray1D arr = fft_alloc_1d(N);
    arr.IsFourier = false; // rare case of real space array
    FFTPlan1D plan = fft_plan_create_1d(N);

    // ----------------------------
    // Initialize array directly on device
    // ----------------------------
    int block = 256;
    int grid = (N + block - 1) / block;
    init_sin_kernel<<<grid, block>>>(arr.d_complex, N, L, kf);
    cudaDeviceSynchronize();
    

    // ----------------------------
    // Forward FFT (real -> complex)
    // ----------------------------
    fft_forward_inplace(plan, arr);

    // ----------------------------
    // Allocate spectrum array on device
    // ----------------------------
    double* d_spectrum;
    cudaMalloc(&d_spectrum, sizeof(double) * N );

    // Compute spectrum |F(k)|^2
    compute_normalized_spectrum(arr, d_spectrum);

    // ----------------------------
    // Copy spectrum back to host
    // ----------------------------
    double* h_spectrum = new double[N];
    cudaMemcpy(h_spectrum, d_spectrum, sizeof(double) * N, 
		     cudaMemcpyDeviceToHost);

    // ----------------------------
    // Write spectrum to file
    // ----------------------------
    std::ofstream fout("spectrum.txt");
    for (int k = 0; k < N; ++k) {
        fout << k << " " << h_spectrum[k] << "\n";
    }
    fout.close();
    delete[] h_spectrum;

    // ----------------------------
    // Clean up
    // ----------------------------
    cudaFree(d_spectrum);
    fft_free_1d(arr);
    fft_plan_destroy_1d(plan);

    std::cout << "Spectrum written to spectrum.txt\n";
    return 0;
}
