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
// Euler update: Y <- Y + k1
__global__ void euler_update_kernel(cufftDoubleComplex* Y,
                                  const cufftDoubleComplex* k1,
				  const double dt,
                                  int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        Y[i].x += dt*k1[i].x ;
        Y[i].y += dt*k1[i].y ;
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
void TimeStep_free_device_memory(TimeStepDeviceData& dev)
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
void ExpRK4(cufftDoubleComplex* d_psi, int N,  double dt, 
		       TimeStepDeviceData& dev)
{
    const int threads = BLOCK_SIZE;
    const int blocks  = (N + threads - 1) / threads;
     tt = 0
    // First transform variable
    // Y = exp(-G*tt)\psi
    exp_transform(dev.d_Y, d_psi, tt, false, N);
    // 1st evaluation of RHS: k1 = f(Y) -> dev.d_k1
    // d_k1 = exp(-G*tt)*NLIN(psi)
    compute_rhs(dev.d_k1, d_psi, tt, N, 1);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    // take 1st step:
    //Ytemp = Y + 0.5*dt*k1 ; tt = tt + dt/2;
    // psi(tt+dt/2) = Ytemp*exp(G* (tt+dt/2) )
    combine_stage_kernel<<<blocks, threads>>>(dev.d_Ytemp, dev.d_Y,
					      dev.d_k1, 0.5*dt, N );
    CUDA_CHECK(cudaDeviceSynchronize());
    tt = dt/2.;
    exp_transform(d_psi, dev.d_Ytemp, -tt, false, N); //actually inverse transform
    // 2nd evaluation of RHS: k2 = f(Ytemp) -> dev.d_k2
    // d_k2 = exp(-G*tt)*NLIN(psi)
    compute_rhs(dev.d_k2, d_psi, tt, N, 2);
    //take 2nd step:
    //Ytemp = Y + 0.5*dt*k2 ; tt = dt/2;
    // psi(dt/2) = Ytemp*exp(G* (dt/2) )
    combine_stage_kernel<<<blocks, threads>>>(dev.d_Ytemp, dev.d_Y,
					      dev.d_k2, 0.5*dt, N );
    CUDA_CHECK(cudaDeviceSynchronize());
    tt = dt/2.;
    exp_transform(d_psi, dev.d_Ytemp, -tt, false, N); //actually inverse transform
    // 3rd evaluation of RHS: k3 = f(Ytemp) -> dev.d_k3
    // d_k3 = exp(-G*tt)*NLIN(psi)
    tt = dt/2
    compute_rhs(dev.d_k2, d_psi, tt, N, 3);
    //take 3rd step:
    //Ytemp = Y + dt*k3 ; tt = dt;
    // psi(dt) = Ytemp*exp(G* dt )
    tt = dt;
    combine_stage_kernel<<<blocks, threads>>>(dev.d_Ytemp, dev.d_Y,
					      dev.d_k3, dt, N );
    CUDA_CHECK(cudaDeviceSynchronize());
    tt = dt;
    exp_transform(d_psi, dev.d_Ytemp, -tt, false, N); //actually inverse transform



    
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
//--------------------------------------------------//
void ExpRK2(cufftDoubleComplex* d_psi, int N,  double dt, 
		       TimeStepDeviceData& dev)
{
    const int threads = BLOCK_SIZE;
    const int blocks  = (N + threads - 1) / threads;
    double tt = 0;
    // First transform variable
    // Y = exp(-G*tt)\psi
    exp_transform(dev.d_Y, d_psi, tt, false, N);
    // 1st evaluation of RHS: k1 = f(Y) -> dev.d_k1
    // d_k1 = exp(-G*tt)*NLIN(psi)
    compute_rhs(dev.d_k1, d_psi, tt, N, 1);
    // take half step:
    //Ytemp = Y + 0.5*dt*k1 ; tt = tt + dt/2;
    // psi(tt+dt/2) = Ytemp*exp(G* (tt+dt/2) )
    combine_stage_kernel<<<blocks, threads>>>(dev.d_Ytemp, dev.d_Y,
					      dev.d_k1, 0.5*dt, N );
    CUDA_CHECK(cudaDeviceSynchronize());
    tt = tt + dt/2.;
    exp_transform(d_psi, dev.d_Ytemp, -tt, false, N); //actually inverse transform
    // 2nd evaluation of RHS: k2 = f(Ytemp) -> dev.d_k2
    // d_k2 = exp(-G*tt)*NLIN(psi)
    compute_rhs(dev.d_k2, d_psi, tt, N, 2);
    // Final update Y <- Y + d_k2*dt
    euler_update_kernel<<<blocks, threads>>>(dev.d_Y, dev.d_k2, dt, N);
    exp_transform(d_psi, dev.d_Y,-dt, false, N); //actually inverse transform
    CUDA_CHECK(cudaDeviceSynchronize());
 }
//---------------------------
void ExpEuler(cufftDoubleComplex* d_psi, int N,  double dt, 
		       TimeStepDeviceData& dev)
{
    const int threads = BLOCK_SIZE;
    const int blocks  = (N + threads - 1) / threads;
    double tt = 0;
    // First transform variable
    // Y = exp(-G*tt)\psi
    exp_transform(dev.d_Y, d_psi, tt, false, N);
    // step one
    // d_k1 = exp(-G*tt)*NLIN(psi)
    compute_rhs(dev.d_k1, d_psi, tt, N, 1);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());     
    // Final update Y <- Y + d_k1*dt
    euler_update_kernel<<<blocks, threads>>>(dev.d_Y, dev.d_k1, dt, N);
    exp_transform(d_psi, dev.d_Y,-dt, false, N); //actually inverse transform
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
 }
//------------------------------------------//
void ExpScheme(cufftDoubleComplex* d_psi, int N,  double dt, 
	       TimeStepDeviceData& dev){
   //ExpEuler(d_psi, N, dt, dev);
   //ExpRK2(d_psi, N, dt, dev);
   ExpRK4(d_psi, N, dt, dev);
}
