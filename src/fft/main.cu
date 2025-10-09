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
        data[i] = sin(50.0 * x);
    }
}

int main() {
    int N = 1024;              // number of real samples
    double L = 2 * M_PI;       // domain size

    // ----------------------------
    // Allocate FFT array and plan
    // ----------------------------
    FFTArray1D arr = fft_alloc_1d(N);
    FFTPlan1D plan = fft_plan_create_1d(N);

    // ----------------------------
    // Initialize array directly on device
    // ----------------------------
    int block = 256;
    int grid = (N + block - 1) / block;
    init_sin_kernel<<<grid, block>>>(arr.d_real, N, L);
    cudaDeviceSynchronize();

    // ----------------------------
    // Forward FFT (real -> complex)
    // ----------------------------
    fft_forward_inplace(plan, arr);

    // ----------------------------
    // Allocate spectrum array on device
    // ----------------------------
    double* d_spectrum;
    cudaMalloc(&d_spectrum, sizeof(double) * (N/2 + 1));

    // Compute spectrum |F(k)|^2
    compute_spectrum(arr, d_spectrum);

    // ----------------------------
    // Copy spectrum back to host
    // ----------------------------
    double* h_spectrum = new double[N/2 + 1];
    cudaMemcpy(h_spectrum, d_spectrum, sizeof(double) * (N/2 + 1), cudaMemcpyDeviceToHost);

    // ----------------------------
    // Write spectrum to file
    // ----------------------------
    std::ofstream fout("spectrum.txt");
    for (int k = 0; k <= N/2; ++k) {
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
