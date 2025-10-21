#include <curand_kernel.h>
#include <math.h>
#include "fft_utils.h"
#include "misc.h"
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error %s:%d: '%s'\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                               \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)
// ******************************************** //
void  double2FFTArray(FFTArray1D& Arr, double* yy, int N){
  Arr.N = N;
  Arr.d_real = yy;
  Arr.d_complex = reinterpret_cast<cufftDoubleComplex*>(Arr.d_real);
}
//--------------------
FFTArray1D fft_alloc_1d(int N) {
    FFTArray1D arr;
    arr.N = N;

    size_t bytes = sizeof(double) * (N + 2);
    CUDA_CHECK(cudaMalloc(&arr.d_real, bytes));
    arr.d_complex = reinterpret_cast<cufftDoubleComplex*>(arr.d_real);

    return arr;
}
//
void fft_free_1d(FFTArray1D& arr) {
  CUDA_CHECK(cudaFree(arr.d_real));
  arr.d_real = nullptr;
  arr.d_complex = nullptr;
}
//
void copy_FFTArray_host_complex(cufftDoubleComplex* h_arr, FFTArray1D& arr){
  int N = arr.N;
  CUDA_CHECK(cudaMemcpy(h_arr, arr.d_complex, 
			  sizeof(cufftDoubleComplex) * (N/2 + 1),
			cudaMemcpyDeviceToHost));
}
void copy_FFTArray_host(double* h_arr, FFTArray1D& arr){
  int N = arr.N;
  CUDA_CHECK(cudaMemcpy(h_arr, arr.d_real, sizeof(double) * (N + 2),
			cudaMemcpyDeviceToHost));
}
//
__global__ void B_Adt_kernel(cufftDoubleComplex* A,
			     cufftDoubleComplex* B,
			     int N,
			     double dt){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N + 2 ) {
	B[i].x = A[i].x * dt;
	B[i].y = A[i].y * dt;
    }
}
//
void B_Adt_FFTArray(FFTArray1D& A, FFTArray1D& B, double dt){
  int block = 256;
  int grid = (A.N + block - 1) / block;
  B_Adt_kernel<<<grid, block>>>(A.d_complex, B.d_complex, A.N, dt);
  
}
__global__ void copy_array_kernel(double* A, double* B, int M){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M ) {
        B[i] = A[i]; 
    }
}
void copy_FFTArray(const FFTArray1D& A, FFTArray1D& B){
  B.N = A.N;
  B.IsFourier = A.IsFourier;
  int block = 256;
  int grid = (A.N + 2 + block - 1) / block;
  copy_array_kernel<<<grid, block>>>(A.d_real, B.d_real, A.N + 2);
}
//
__global__ void cube_array_kernel(double* A, int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N ) {
      double x = A[i];
      A[i] = x * x * x; 
    }
    if (i == N || i == N + 1){A[i] = 0.;}
}
//
void cube_FFTArray(FFTArray1D& A){
  if(A.IsFourier){
      clean_exit_host("cub_FFTArray only in real space", 0);
  }else{
    int block = 256;
    int grid = (A.N + 2 + block - 1) / block;
    cube_array_kernel<<<grid, block>>>(A.d_real, A.N );
  }
}
//
__global__ void quartic_array_kernel(double* A, int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N ) {
        A[i] = A[i]*A[i]*A[i]*A[i]; 
    }
}
//
void quartic_FFTArray(FFTArray1D& A){
  int block = 256;
  int grid = (A.N + block - 1) / block;
  quartic_array_kernel<<<grid, block>>>(A.d_real, A.N );
}
// Forward FFT (real → complex)
void fft_forward_inplace(const FFTPlan1D& plan, FFTArray1D& arr) {
    if (arr.IsFourier){
      clean_exit_host("fft_forward_inplace : wrong call", 0);
    }else{
    cufftExecD2Z(plan.plan_fwd,
                 reinterpret_cast<cufftDoubleReal*>(arr.d_real),
                 reinterpret_cast<cufftDoubleComplex*>(arr.d_complex));
    arr.IsFourier = true;
    }
}
// Inverse FFT (complex → real)
void fft_inverse_inplace(const FFTPlan1D& plan, FFTArray1D& arr) {
    if (arr.IsFourier){
      cufftExecZ2D(plan.plan_inv,
                 reinterpret_cast<cufftDoubleComplex*>(arr.d_complex),
                 reinterpret_cast<cufftDoubleReal*>(arr.d_real));
      arr.IsFourier = false;
    }else{
      clean_exit_host("fft_inverse_inplace : wrong call", 0);
    }

}
//
__global__ void normalize_fft_kernel(double* data, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        data[i] /= N;  // scale inverse FFT
    }
    if (i == N || i == N + 1){
       data[i] = 0.; //make sure zero-padding.
    }
}
//
void normalize_fft(FFTArray1D& arr) {
    if (arr.IsFourier){
      clean_exit_host("normalize_fft works only for real array", 0);
    }else{
      int block = 256;
      int grid = (arr.N + 2 + block - 1) / block;
      normalize_fft_kernel<<<grid, block>>>(arr.d_real, arr.N);
      cudaDeviceSynchronize();
    }
}

//spectrum calculation 
__global__ void compute_spectrum_kernel(const cufftDoubleComplex* data,
                             double* spectrum,
                             int nfreqs)
{
     int i = blockIdx.x * blockDim.x + threadIdx.x;
     if (i < nfreqs) {
       double re = data[i].x;
       double im = data[i].y;
       spectrum[i] = re * re + im * im;  // |F(k)|²
     }
}
//
void compute_spectrum(const FFTArray1D& arr, double* d_spectrum)
{
    if(arr.IsFourier){
      int nfreqs = arr.N / 2 + 1;
      int block = 256;
      int grid = (nfreqs + block - 1) / block;
      compute_spectrum_kernel<<<grid, block>>>(arr.d_complex,
					       d_spectrum, nfreqs);
    }else{
      clean_exit_host("compute_spectrum works only in fourier space", 0);
    }
}

__global__ void normalize_spectrum_kernel(double* spectrum, int nfreqs, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nfreqs) {
        spectrum[i] /= (N * N); // normalize |F(k)|^2
    }
}

void normalize_spectrum(double* d_spectrum, int N) {
    int nfreqs = N / 2 + 1;
    int block = 256;
    int grid = (nfreqs + block - 1) / block;
    normalize_spectrum_kernel<<<grid, block>>>(d_spectrum, nfreqs, N);
    cudaDeviceSynchronize();
}
// Generates data in fourier space with a given spectrum

// Kernel to set power-law spectrum using Mersenne Twister RNG
__global__ void power_law_spectrum_kernel(cufftDoubleComplex* data,
                                             int N,
                                             double A,
                                             double xi,
                                             int kmin,
                                             int kmax,
                                             unsigned long seed)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int nfreqs = N / 2 + 1;
    if (i >= nfreqs) return;

    double re = 0.0;
    double im = 0.0;

    if (i >= kmin && i <= kmax) {
        // 
	curandStatePhilox4_32_10_t state;
        curand_init(seed + i, 0, 0, &state); 

        // Generate uniform random phase [0, 2pi)
        double phi = curand_uniform_double(&state) * 2.0 * M_PI;

        // Amplitude such that |F(k)|^2 = A k^xi
        double amplitude = sqrt(A * pow((double)i, xi));
        re = amplitude * cos(phi);
        im = amplitude * sin(phi);
    }

    data[i].x = re;
    data[i].y = im;
    if ( i == 0 || i == N/2 ) {data[i].y = 0.;}
}
//------------------------
void set_power_law_spectrum(FFTArray1D& arr,
                                      double A, double xi,
                                      int kmin, int kmax,
                                      unsigned long seed = 1234)
{
    if (arr.IsFourier){
      int nfreqs = arr.N / 2 + 1;
      int block = 256;
      int grid = (nfreqs + block - 1) / block;

      power_law_spectrum_kernel<<<grid, block>>>(arr.d_complex,
                                                  arr.N,
                                                  A, xi, kmin, kmax,
                                                  seed);
      cudaDeviceSynchronize();
    }else{
      clean_exit_host("set power law spectrum work in fourier space", 0);
    }
}
// Kernel to set power-law spectrum using Mersenne Twister RNG
__global__ void peak_spectrum_kernel(cufftDoubleComplex* data,
		                    int N,
                                    double A, double dk,
                                    int kf,
                                    unsigned long seed,
				    bool sharp)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int nfreqs = N / 2 + 1;
    if (i >= nfreqs) return;

    double re = 0.0;
    double im = 0.0;

   // 
   curandStatePhilox4_32_10_t state;
   curand_init(seed + i, 0, 0, &state); 

   // Generate uniform random phase [0, 2pi)
   double phi = curand_uniform_double(&state) * 2.0 * M_PI;

  // Amplitude such that |F(k)|^2 = A k^xi
   if (sharp){
     if(i == kf){
        re = 0 ;
	im = A ;
        data[i].x = re;
        data[i].y = im;	
     }
   }else{
   double KF = (double) kf;
   double kk = (double) i;
   double amplitude = gaussian(kk, kf, 2*dk, A);

   re = amplitude * cos(phi);
   im = amplitude * sin(phi);

    data[i].x = re;
    data[i].y = im;
   }
   if ( i == 0 || i == N/2 ) {data[i].y = 0.;}
}
//--------------------------------
void set_peak_spectrum(FFTArray1D& arr,
                                 double A, double dk,
                                 int kf, 
                                 unsigned long seed = 1234,
				 bool sharp = 0)
{ 
    if (arr.IsFourier){
      int nfreqs = arr.N / 2 + 1;
      int block = 256;
      int grid = (nfreqs + block - 1) / block;

      peak_spectrum_kernel<<<grid, block>>>(arr.d_complex,
                                          arr.N,
                                          A, dk, kf, seed, sharp);
      cudaDeviceSynchronize();
    }else{
      clean_exit_host("set_peak_spectrum works only in fourier space", 0);
    }
}
// ---------------------
__global__ void set_zero_kernel(double* data, int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N+2) return;
    data[i] = 0.0;
}
//----------------------------------------
void set_zero(FFTArray1D& arr){
    int N = arr.N;
    int block = 256;
    int grid = (N + 2+ block - 1) / block; //+2 because the array is N+2

    set_zero_kernel<<<grid, block>>>(arr.d_real, arr.N);
    cudaDeviceSynchronize();
}
//---------------
__global__ void  complex_mult_kernel(cufftDoubleComplex* data,
				     cufftDoubleComplex z, int N){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int nfreqs = N / 2 + 1;
  if (i >= nfreqs) return;
  data[i] = cuCmul(data[i],z);  
}
//----------------------------
void  complex_mult_FFTArray(FFTArray1D& arr, cufftDoubleComplex z){
  int nfreqs = arr.N / 2 + 1;
  int block = 256;
  int grid = (nfreqs + block - 1) / block;
  complex_mult_kernel<<<grid, block>>>(arr.d_complex, z, arr.N);
}
//
__global__ void derivk_kernel(cufftDoubleComplex* data, double hh, int N,
			      bool abs){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int nfreqs = N / 2 + 1;
    if (i >= nfreqs) return;
    if (i == 0) return;

   //
   double kk = (double) i;
   if (abs){
     data[i].x = pow(kk,hh)*data[i].x ; 
     data[i].y = pow(kk,hh)*data[i].y ;
   }else{ 
     cufftDoubleComplex Ikk;
     Ikk.x  = 0; Ikk.y = kk;
     cufftDoubleComplex IIkh = cuCpow(Ikk,hh);
     cufftDoubleComplex fk;
     fk.x = data[i].x;
     fk.y = data[i].y;
     cufftDoubleComplex dh_f = cuCmul(IIkh,fk);
     data[i].x = dh_f.x; 
     data[i].y = dh_f.y; 
   }
}
//----------------------------------------
void derivk(FFTArray1D& arr, double hh,  bool abs){
	// calculates hh order derivative of array ff in Fourier space 
	// hh can be fractional including negative. The output is stored
	// in ff itself. If abs is set then the absolute value of the 
       // derivative is calculated. 	
    if (arr.IsFourier){
      int nfreqs = arr.N / 2 + 1;
      int block = 256;
      int grid = (nfreqs + block - 1) / block;
      derivk_kernel<<<grid, block>>>(arr.d_complex, hh, arr.N, abs);
    }else{
      clean_exit_host("derivk works only in fourier space", 0);
    }
}
//
__global__ void sine_kernel(double* data, double dx, double A, int kpeak, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        double x = i * dx;
	double KPEAK = (double) kpeak;
        data[i] = A*sin(KPEAK * x);
    }
}
void set_sine_real(double* data, double dx, double A, int kpeak, int N){
    int block = 256;
    int grid = (N + block - 1) / block;
    sine_kernel<<<grid, block>>>(data, dx, A, kpeak, N);
}
__global__ void cosine_kernel(double* data, int N, double dx, double A, int kpeak) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        double x = i * dx;
	double KPEAK = (double) kpeak;
        data[i] = A*cos(KPEAK * x);

    }
}
void set_cosine_real(double* data, double dx, double A, int kpeak, int N){
    int block = 256;
    int grid = (N + block - 1) / block;
    cosine_kernel<<<grid, block>>>(data, N, dx, A, kpeak);
}
