#include <curand_kernel.h>
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
void copy_FFTArray(FFTArray1D& A, FFTArray1D& B){
  B.N = A.N;
  int block = 256;
  int grid = (A.N + block - 1) / block;
  copy_array_kernel<<<grid, block>>>(A.d_real, B.d_real, A.N + 2);
}
//
__global__ void cube_array_kernel(double* A, int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N ) {
        A[i] = A[i]*A[i]*A[i]; 
    }
}
//
void cube_FFTArray(FFTArray1D& A){
  int block = 256;
  int grid = (A.N + block - 1) / block;
  cube_array_kernel<<<grid, block>>>(A.d_real, A.N );
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
  cube_array_kernel<<<grid, block>>>(A.d_real, A.N );
}
// Forward FFT (real → complex)
void fft_forward_inplace(const FFTPlan1D& plan, FFTArray1D& arr) {
    cufftExecD2Z(plan.plan_fwd,
                 reinterpret_cast<cufftDoubleReal*>(arr.d_real),
                 reinterpret_cast<cufftDoubleComplex*>(arr.d_complex));
}

// Inverse FFT (complex → real)
void fft_inverse_inplace(const FFTPlan1D& plan, FFTArray1D& arr) {
    cufftExecZ2D(plan.plan_inv,
                 reinterpret_cast<cufftDoubleComplex*>(arr.d_complex),
                 reinterpret_cast<cufftDoubleReal*>(arr.d_real));
}
//
__global__ void normalize_fft_kernel(double* data, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        data[i] /= N;  // scale inverse FFT
    }
}
//
void normalize_fft(FFTArray1D& arr) {
    int block = 256;
    int grid = (arr.N + block - 1) / block;
    normalize_fft_kernel<<<grid, block>>>(arr.d_real, arr.N);
    cudaDeviceSynchronize();
}

//spectrum calculation 
__global__
void compute_spectrum_kernel(const cufftDoubleComplex* data,
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

void compute_spectrum(const FFTArray1D& arr, double* d_spectrum) {
    int nfreqs = arr.N / 2 + 1;
    int block = 256;
    int grid = (nfreqs + block - 1) / block;
    compute_spectrum_kernel<<<grid, block>>>(arr.d_complex, d_spectrum, nfreqs);
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
}
//------------------------
void set_power_law_spectrum(FFTArray1D& arr,
                                      double A, double xi,
                                      int kmin, int kmax,
                                      unsigned long seed = 1234)
{
    int nfreqs = arr.N / 2 + 1;
    int block = 256;
    int grid = (nfreqs + block - 1) / block;

    power_law_spectrum_kernel<<<grid, block>>>(arr.d_complex,
                                                  arr.N,
                                                  A, xi, kmin, kmax,
                                                  seed);
    cudaDeviceSynchronize();
}
// Kernel to set power-law spectrum using Mersenne Twister RNG
__global__ void peak_spectrum_kernel(cufftDoubleComplex* data,
		                    int N,
                                    double A, double dk,
                                    int kf, 
                                    unsigned long seed)
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
   double KF = (double) kf;
   double kk = (double) i;
   double amplitude = gaussian(kk, kf, 2*dk, A);

   re = amplitude * cos(phi);
   im = amplitude * sin(phi);

    data[i].x = re;
    data[i].y = im;
}
//--------------------------------
void set_peak_spectrum(FFTArray1D& arr,
                                 double A, double dk,
                                 int kf, 
                                 unsigned long seed = 1234)
{
    int nfreqs = arr.N / 2 + 1;
    int block = 256;
    int grid = (nfreqs + block - 1) / block;

    peak_spectrum_kernel<<<grid, block>>>(arr.d_complex,
                                                  arr.N,
                                                  A, dk, kf, seed);
    cudaDeviceSynchronize();
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
__global__ void derivk_kernel(cufftDoubleComplex* data, double hh, int N, bool abs){
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
    int nfreqs = arr.N / 2 + 1;
    int block = 256;
    int grid = (nfreqs + block - 1) / block;
    derivk_kernel<<<grid, block>>>(arr.d_complex, hh, arr.N, abs);
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
