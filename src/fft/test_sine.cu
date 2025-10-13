#include <iostream>
#include <fstream>
#include <cmath>
#include <cuda_runtime.h>
#include "fft_utils.h"
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error %s:%d: '%s'\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                               \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

/***************************************************************/
int main(int argc, char** argv){
    int N = 1024;              // number of real samples
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
    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    printf("GPU memory: %.2f / %.2f GB free\n", 
		 (double)free_mem/1000000000.0, 
		 (double)total_mem/1000000000.0);
    FFTArray1D arr = fft_alloc_1d(N);
    FFTPlan1D plan = fft_plan_create_1d(N);
    // ----------------------------
    // Initialize array as sine in real space 
    // ----------------------------
    double dx = L/(double) N ;
    set_sine_real(arr.d_real, dx, 1., 1, N);
    // copy back to host
    double* f = new double[N + 2];
    copy_FFTArray_host(f, arr);
    //CUDA_CHECK(cudaMemcpy(f, arr.d_real, sizeof(double) * (N + 2),
    //			  cudaMemcpyDeviceToHost));
    // transform to Fourier space
    fft_forward_inplace(plan, arr);
    // ----------------------------
    // Allocate spectrum array on device
    // ----------------------------
    double* d_spectrum;
    CUDA_CHECK(cudaMalloc(&d_spectrum, sizeof(double) * (N/2 + 1)));
    // Compute spectrum |F(k)|^2
    compute_spectrum(arr, d_spectrum);
    // ----------------------------
    // Copy spectrum back to host
    // ----------------------------
    double* h_spectrum = new double[N/2 + 1];
    cudaMemcpy(h_spectrum, d_spectrum, sizeof(double) * (N/2 + 1),
	       cudaMemcpyDeviceToHost);
    // ----------------------------
    // Write spectrum to file
    // ----------------------------
    std::ofstream sout("spectrum.txt");
    for (int k = 0; k <= N/2; ++k) {
        sout << k << " " << h_spectrum[k] << "\n";
    }
    sout.close();
    delete[] h_spectrum;

    // ----------------------------
    // Write data file
    // ----------------------------
    std::ofstream fout("data.txt");
    for (int i = 0; i < N; ++i) {
      double x = i * dx;
      fout << x << " " << f[i] << " " << sin(x) << "\n";
    }
    fout.close();
    // ----------------------------
    // Clean up
    // ----------------------------
    cudaFree(d_spectrum);
    fft_free_1d(arr);
    fft_plan_destroy_1d(plan);
    delete[] f;
    std::cout << "Spectrum written to spectrum.txt\n";
    return 0;
}
