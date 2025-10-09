#ifndef HARMONIC_H
#define HARMONIC_H

#include <cuda_runtime.h>

// Structure for harmonic oscillator parameters
struct HarmonicParams {
    double m;
    double k;
};

// ------------------------------------------------------
// Device RHS kernel
// ------------------------------------------------------
__global__ void rhs_kernel(const double* Y, double* out, double dt, int N);

// ------------------------------------------------------
// Device energy kernel
// ------------------------------------------------------
__global__ void compute_energy_kernel(const double* Y, double* d_E);

// Host wrapper for energy computation
inline void compute_energy_device(const double* d_Y, double* d_E) {
    compute_energy_kernel<<<1,1>>>(d_Y, d_E);
    cudaDeviceSynchronize();
}

// ------------------------------------------------------
// Host helper to copy parameters to device constant memory
// ------------------------------------------------------
void copy_params_to_device(const HarmonicParams& h_params);

#endif // HARMONIC_H

