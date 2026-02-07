// rk4.cu
#include <cstdio>
#include "evolve.h"
#include "model.h"   // declaration of rhs_kernel 

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
__global__ void combine_stage_kernel(cufftDoubleComplex* Ytemp,
                                     const cufftDoubleComplex* Y,
                                     const cufftDoubleComplex* k,
                                     double a,
                                     int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
	    Ytemp[i].x = Y[i].x + a * k[i].x;
	    Ytemp[i].y = Y[i].y + a * k[i].y;
    }
}

// final RK4 update: Y <- Y + (k1 + 2*k2 + 2*k3 + k4)/6
__global__ void rk4_update_kernel(cufftDoubleComplex* Y,
                                  const cufftDoubleComplex* k1,
                                  const cufftDoubleComplex* k2,
                                  const cufftDoubleComplex* k3,
                                  const cufftDoubleComplex* k4,
                                  int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        Y[i].x += (k1[i].x + 2.0 * k2[i].x + 2.0 * k3[i].x + k4[i].x) / 6.0;
        Y[i].y += (k1[i].y + 2.0 * k2[i].y + 2.0 * k3[i].y + k4[i].y) / 6.0;
    }
}
// ---------------- host functions ----------------
TimeStepDeviceData TimeStep_allocate_device_memory(int N)
{
    TimeStepDeviceData dev;
    dev.is_initialized = false;

    CUDA_CHECK(cudaMalloc(&dev.d_Y, N * sizeof(cufftDoubleComplex) ));
    CUDA_CHECK(cudaMalloc(&dev.d_Ytemp, N * sizeof(cufftDoubleComplex) ));
    CUDA_CHECK(cudaMalloc(&dev.d_k1,    N * sizeof(cufftDoubleComplex) ));
    CUDA_CHECK(cudaMalloc(&dev.d_k2,    N * sizeof(cufftDoubleComplex) ));
    CUDA_CHECK(cudaMalloc(&dev.d_k3,    N * sizeof(cufftDoubleComplex) ));
    CUDA_CHECK(cudaMalloc(&dev.d_k4,    N * sizeof(cufftDoubleComplex) ));

    return dev;
}

void rk4_free_device_memory(TimeStepDeviceData& dev)
{

    if (dev.d_Y) cudaFree(dev.d_Y);
    if (dev.d_Ytemp) cudaFree(dev.d_Ytemp);
    if (dev.d_k1)    cudaFree(dev.d_k1);
    if (dev.d_k2)    cudaFree(dev.d_k2);
    if (dev.d_k3)    cudaFree(dev.d_k3);
    if (dev.d_k4)    cudaFree(dev.d_k4);

    dev.d_Y = dev.d_Ytemp = dev.d_k1 = dev.d_k2 = dev.d_k3 = dev.d_k4 = nullptr;
    dev.is_initialized = false;
}

//-----------------------------------//
void ExpScheme(cufftDoubleComplex* d_psi, int N,  double dt, 
		       TimeStepDeviceData& dev)
{
    if (N <= 0) return;
    const int threads = BLOCK_SIZE;
    const int blocks  = (N + threads - 1) / threads;
    double tt = 0;
    // First transform variable
    exp_transform(dev.d_Y, d_psi, tt, false, N);
    // Stage 1: k1 = dt * f(Y) -> dev.d_k1
    compute_rhs(dev.d_k1, d_psi, tt, N, 1);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Stage 2: Ytemp = Y + 0.5*k1 ; k2 = dt * f(Ytemp)
    combine_stage_kernel<<<blocks, threads>>>(dev.d_Ytemp,
					      dev.d_Y, dev.d_k1, 0.5, N );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    tt = dt/2.;
    exp_transform(d_psi, dev.d_Ytemp, -tt, false, N); //actually inverse transform
    compute_rhs(dev.d_k2, d_psi, tt, N, 2);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Stage 3: Ytemp = Y + 0.5*k2 ; k3 = dt * f(Ytemp)
    combine_stage_kernel<<<blocks, threads>>>(dev.d_Ytemp,
					      dev.d_Y, dev.d_k2, 0.5, N );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    tt = dt/2.;
    exp_transform(d_psi, dev.d_Ytemp, -tt, false, N); //actually inverse transform
    compute_rhs(dev.d_k3, d_psi, tt, N, 3);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Stage 4: Ytemp = Y + k3 ; k4 = dt * f(Ytemp)
    combine_stage_kernel<<<blocks, threads>>>(dev.d_Ytemp,
					      dev.d_Y, dev.d_k3, 1.0, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    tt = dt;
    exp_transform(d_psi, dev.d_Ytemp, -tt, false, N); //actually inverse transform
    compute_rhs(dev.d_k4, d_psi, tt, N, 4);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
      
    // Final update Y <- Y + (k1 + 2*k2 + 2*k3 + k4) / 6
    rk4_update_kernel<<<blocks, threads>>>(dev.d_Y,
					   dev.d_k1, dev.d_k2,
					   dev.d_k3, dev.d_k4, N);
    exp_transform(d_psi, dev.d_Y, -dt, false, N); //actually inverse transform
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
}
