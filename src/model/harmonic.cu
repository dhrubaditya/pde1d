#include "harmonic.h"

// Define the device constant variable only once
__constant__ HarmonicParams d_params;

// ------------------------------------------------------
// Host function to copy parameters to device constant memory
// ------------------------------------------------------
void copy_params_to_device(const HarmonicParams& h_params)
{
    cudaMemcpyToSymbol(d_params, &h_params, sizeof(HarmonicParams));
}

// ------------------------------------------------------
// RHS kernel
// ------------------------------------------------------
__global__ void rhs_kernel(const double* Y, double* out, double dt, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    double m = d_params.m;
    double k = d_params.k;

    if (i == 0) {
        out[0] = dt * Y[1];
    } else if (i == 1) {
        out[1] = dt * (-(k / m) * Y[0]);
    } else {
        out[i] = 0.0;
    }
}

// ------------------------------------------------------
// Energy kernel
// ------------------------------------------------------
__global__ void compute_energy_kernel(const double* Y, double* d_E)
{
    double m = d_params.m;
    double k = d_params.k;
    double x = Y[0];
    double v = Y[1];
    *d_E = 0.5 * m * v * v + 0.5 * k * x * x;
}

