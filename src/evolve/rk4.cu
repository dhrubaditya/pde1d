// rk4.cu
#include "rk4.h"
#include "harmonic.h"   // declaration of rhs_kernel and energy kernel wrapper
#include <cstdio>

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error %s:%d: '%s'\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                               \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

static const int BLOCK_SIZE = RK4_BLOCK_SIZE;

// ---------------- kernels ----------------
// combine stage: Ytemp = Y + a * k
__global__ void combine_stage_kernel(double* Ytemp,
                                     const double* Y,
                                     const double* k,
                                     double a,
                                     int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) Ytemp[i] = Y[i] + a * k[i];
}

// final RK4 update: Y <- Y + (k1 + 2*k2 + 2*k3 + k4)/6
__global__ void rk4_update_kernel(double* Y,
                                  const double* k1,
                                  const double* k2,
                                  const double* k3,
                                  const double* k4,
                                  int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        Y[i] += (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0;
    }
}

// ---------------- host functions ----------------
RK4DeviceData rk4_allocate_device_memory(int N)
{
    RK4DeviceData dev;
    dev.is_initialized = false;

    CUDA_CHECK(cudaMalloc(&dev.d_Y,     N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dev.d_Ytemp, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dev.d_k1,    N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dev.d_k2,    N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dev.d_k3,    N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dev.d_k4,    N * sizeof(double)));

    return dev;
}

void rk4_free_device_memory(RK4DeviceData& dev)
{
    if (dev.d_Y)     cudaFree(dev.d_Y);
    if (dev.d_Ytemp) cudaFree(dev.d_Ytemp);
    if (dev.d_k1)    cudaFree(dev.d_k1);
    if (dev.d_k2)    cudaFree(dev.d_k2);
    if (dev.d_k3)    cudaFree(dev.d_k3);
    if (dev.d_k4)    cudaFree(dev.d_k4);

    dev.d_Y = dev.d_Ytemp = dev.d_k1 = dev.d_k2 = dev.d_k3 = dev.d_k4 = nullptr;
    dev.is_initialized = false;
}

// Perform M RK4 timesteps. The RHS for each stage is computed by a single
// device-parallel kernel rhs_kernel which writes dt*f(Y) into the k arrays.
void rk4_timestep_host(double* Y_host, int N, double dt, int M, RK4DeviceData& dev)
{
    if (N <= 0) return;
    const int threads = BLOCK_SIZE;
    const int blocks  = (N + threads - 1) / threads;

    // Copy host->device only on first call
    if (!dev.is_initialized) {
        CUDA_CHECK(cudaMemcpy(dev.d_Y, Y_host, N * sizeof(double), cudaMemcpyHostToDevice));
        dev.is_initialized = true;
    }

    for (int step = 0; step < M; ++step) {
        // Stage 1: k1 = dt * f(Y) -> dev.d_k1
        rhs_kernel<<<blocks, threads>>>(dev.d_Y, dev.d_k1, dt, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Stage 2: Ytemp = Y + 0.5*k1 ; k2 = dt * f(Ytemp)
        combine_stage_kernel<<<blocks, threads>>>(dev.d_Ytemp, dev.d_Y, dev.d_k1, 0.5, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        rhs_kernel<<<blocks, threads>>>(dev.d_Ytemp, dev.d_k2, dt, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Stage 3: Ytemp = Y + 0.5*k2 ; k3 = dt * f(Ytemp)
        combine_stage_kernel<<<blocks, threads>>>(dev.d_Ytemp, dev.d_Y, dev.d_k2, 0.5, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        rhs_kernel<<<blocks, threads>>>(dev.d_Ytemp, dev.d_k3, dt, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Stage 4: Ytemp = Y + k3 ; k4 = dt * f(Ytemp)
        combine_stage_kernel<<<blocks, threads>>>(dev.d_Ytemp, dev.d_Y, dev.d_k3, 1.0, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        rhs_kernel<<<blocks, threads>>>(dev.d_Ytemp, dev.d_k4, dt, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Final update Y <- Y + (k1 + 2*k2 + 2*k3 + k4) / 6
        rk4_update_kernel<<<blocks, threads>>>(dev.d_Y, dev.d_k1, dev.d_k2, dev.d_k3, dev.d_k4, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Copy updated solution back to host
    CUDA_CHECK(cudaMemcpy(Y_host, dev.d_Y, N * sizeof(double), cudaMemcpyDeviceToHost));
}

