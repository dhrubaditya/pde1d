#include <iostream>
#include <fstream>
#include <cmath>
#include <cuda_runtime.h>
#include "fft_utils.h"


int main() {
    int N = 1024;              // number of real samples
    double L = 2 * M_PI;       // domain size
    // ----------------------------
    // Allocate FFT array and plan
    // ----------------------------
    FFTArray1D arr = fft_alloc_1d(N);
    FFTPlan1D plan = fft_plan_create_1d(N);

    // ----------------------------
    // Initialize array as a power law spectrum 
    // ----------------------------
    double A = 1.0;
    double xi = 2.0;
    double kmin = 4;
    double kmax= 32;
    double dk = 1.;
    unsigned long long seed = static_cast<unsigned long long>(time(nullptr));
//    set_power_law_spectrum(arr, A, xi, kmin, kmax, seed);
    int kf = 32;
    set_peak_spectrum(arr, A, dk, kf, seed);
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
