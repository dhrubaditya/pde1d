#include "fft_utils.h"

#include <iostream>
#include <fstream>
#include <cmath>
#include <cuda_runtime.h>

// Kernel to initialize real-space array: f(x) = sin(5x)
__global__ void init_sin_kernel(double* data, int N, double L, double kf) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        double dx = L / N;
        double x = i * dx;
        data[i] = sin(x * (double) kf);
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
    FFTPlan1D plan = fft_plan_create_1d(N);

    // ----------------------------
    // Initialize array directly on device in real space
    // ----------------------------
    arr.IsFourier = false;
    int block = 256;
    int grid = (N + block - 1) / block;
    init_sin_kernel<<<grid, block>>>(arr.d_real, N, L, kf);
    cudaDeviceSynchronize();
    // copy to host
    double* f = new double[N + 2];
    cudaMemcpy(f, arr.d_real, sizeof(double) * (N + 2), cudaMemcpyDeviceToHost);
    cube_FFTArray(arr); // this is sin^3 now
    // ----------------------------
    // Forward FFT (real -> complex)
    // ----------------------------
    fft_forward_inplace(plan, arr);
    // ----------------------------
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
