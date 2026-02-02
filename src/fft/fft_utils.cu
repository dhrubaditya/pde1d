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
//--------------------
FFTArray1D fft_alloc_1d(int N) {
    FFTArray1D arr;
    arr.N = N;

    size_t bytes = sizeof(cufftDoubleComplex) * N ;
    CUDA_CHECK(cudaMalloc(&arr.d_complex, bytes));

    return arr;
}
//
void fft_free_1d(FFTArray1D& arr) {
  CUDA_CHECK(cudaFree(arr.d_complex));
  arr.d_complex = nullptr;
}
//
void copy_FFTArray_host_complex(cufftDoubleComplex* h_arr, FFTArray1D& arr){
  int N = arr.N;
  CUDA_CHECK( cudaMemcpy(h_arr, arr.d_complex, 
			 sizeof(cufftDoubleComplex) * N,
			 cudaMemcpyDeviceToHost) );
}
//
__global__ void complex2fft_kernel(cufftDoubleComplex* A,
			           cufftDoubleComplex* B,
			           int N )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N  ) {
	B[i].x = A[i].x ;
	B[i].y = A[i].y ;
    }
}
//-------------------------//
void complex2FFTArray(FFTArray1D& Arr, 
		      cufftDoubleComplex* yy, 
		      int N, bool is_fourier)
{
  Arr.N = N;
  int block = 256;
  int grid = (N + block - 1) / block;
  complex2fft_kernel<<<grid, block>>>(Arr.d_complex, yy, N );
}
//
__global__ void B_Adt_kernel(cufftDoubleComplex* A,
			     cufftDoubleComplex* B,
			     int N,
			     double dt){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N  ) {
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
__global__ void copy_array_kernel(cufftDoubleComplex* A, 
		                  cufftDoubleComplex* B, int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N ) {
        B[i].x = A[i].x; 
        B[i].y = A[i].y ; 
    }
}
void copy_FFTArray(const FFTArray1D& A, FFTArray1D& B){
  B.N = A.N;
  B.IsFourier = A.IsFourier;
  int block = 256;
  int grid = (A.N + 2 + block - 1) / block;
  copy_array_kernel<<<grid, block>>>(A.d_complex, B.d_complex, A.N );
}
//
__global__ void abs2_times_z_kernel(cufftDoubleComplex* A, int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N ) {
      double re = A[i].x ;
      double im = A[i].y ;
      double Abs2 = re * re + im * im ;
      A[i].x = Abs2 * A[i].x ; 
      A[i].y = Abs2 * A[i].y ; 
    }
}
//
void abs2_times_z_FFTArray(FFTArray1D& A){
  if(A.IsFourier){
      clean_exit_host("abs_time_z_FFTArray only in real space", 0);
  }else{
    int block = 256;
    int grid = (A.N + block - 1) / block;
    abs2_times_z_kernel<<<grid, block>>>(A.d_complex, A.N );
  }
}
//
__global__ void quartic_array_kernel(cufftDoubleComplex* A, 
		                     double* B, int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N ) {
      double re = A[i].x ;
      double im = A[i].y ;
      double Abs = sqrt( re * re + im * im );
        B[i] = Abs * Abs; 
    }
}
//
void quartic_FFTArray(FFTArray1D& A, double* B){
  if(A.IsFourier){
      clean_exit_host("quartic_FFTArray only in real space", 0);
  }else{
    int block = 256;
    int grid = (A.N + block - 1) / block;
    quartic_array_kernel<<<grid, block>>>(A.d_complex, B, A.N );
  }
}
// Forward FFT (real → complex)
void fft_forward_inplace(const FFTPlan1D& plan, FFTArray1D& arr) {
    if (arr.IsFourier){
      clean_exit_host("fft_forward_inplace : wrong call", 0);
    }else{
    cufftExecZ2Z(plan.plan_fwd,
                 arr.d_complex,
                 arr.d_complex, CUFFT_FORWARD);
    arr.IsFourier = true;
    }
}
// Inverse FFT (complex → real)
void fft_inverse_inplace(const FFTPlan1D& plan, FFTArray1D& arr) {
    if (arr.IsFourier){
      cufftExecZ2Z(plan.plan_inv,
                   arr.d_complex,
                   arr.d_complex, CUFFT_INVERSE );
      arr.IsFourier = false;
    }else{
      clean_exit_host("fft_inverse_inplace : wrong call", 0);
    }

}
//
__global__ void normalize_fft_kernel(cufftDoubleComplex* data, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        data[i].x /= N;  // scale inverse FFT
        data[i].y /= N;  // scale inverse FFT
    }
}
//
void normalize_fft(FFTArray1D& arr) {
    if (arr.IsFourier){
      clean_exit_host("normalize_fft works only in real space", 0);
    }else{
      int block = 256;
      int grid = (arr.N + block - 1) / block;
      normalize_fft_kernel<<<grid, block>>>(arr.d_complex, arr.N);
      cudaDeviceSynchronize();
    }
}

//spectrum calculation 
__device__ __host__ int fft_freq(int i, int N)
{
    return (i < N/2) ? i : i - N ;
}
void test_fft_freq(int N)
{
     for (int i = 0; i< N; i++){
       int j = fft_freq(i, N);
       std::cout << i << " " << j << " " << N <<"\n";
     }
}
//
__global__ void normalized_spectrum_kernel(const cufftDoubleComplex* data,
                             double* spectrum,
                             int N)
{
     int i = blockIdx.x * blockDim.x + threadIdx.x;
     if (i < N) {
       double re = data[i].x;
       double im = data[i].y;
       int ifreq = fft_freq(i, N) ;
       int ik = abs(ifreq);
       spectrum[ik] = (re * re + im * im ) / (N * N) ;  // |F(k)|²
     }
}
//
void compute_normalized_spectrum(const FFTArray1D& arr, double* d_spectrum)
{
    if(arr.IsFourier){
      int N = arr.N ; 
      int block = 256;
      int grid = (N + block - 1) / block;
      normalized_spectrum_kernel<<<grid, block>>>(arr.d_complex,
					       d_spectrum, N);
    }else{
      clean_exit_host("compute_spectrum works only in fourier space", 0);
    }
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
    if (i >= N) return;

    double re = 0.0;
    double im = 0.0;
    int ifreq = fft_freq(i, N) ;
    int ik = abs(ifreq);
    double kk = (double) ik;
    if (ik >= kmin && ik <= kmax) {
        // 
	curandStatePhilox4_32_10_t state;
        curand_init(seed + i, 0, 0, &state); 

        // Generate uniform random phase [0, 2pi)
        double phi = curand_uniform_double(&state) * 2.0 * M_PI;

        // Amplitude such that |F(k)|^2 = A k^xi
        double amplitude = sqrt(A * pow(kk, xi));
        re = amplitude * cos(phi);
        im = amplitude * sin(phi);
    }

    data[i].x = re;
    data[i].y = im;
    if ( i == 0 ) {data[i].x = 0.; data[i].y = 0.;}
}
//------------------------
void set_power_law_spectrum(FFTArray1D& arr,
                                      double A, double xi,
                                      int kmin, int kmax,
                                      unsigned long seed = 1234)
{
    if (arr.IsFourier){
      int N = arr.N ;
      int block = 256;
      int grid = (N + block - 1) / block;

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
    if (i >= N) return;
    int ifreq = fft_freq(i, N) ;
    int ik = abs(ifreq);
    double re = 0.0;
    double im = 0.0;

   // 
   curandStatePhilox4_32_10_t state;
   curand_init(seed + i, 0, 0, &state); 

   // Generate uniform random phase [0, 2pi)
   double phi = curand_uniform_double(&state) * 2.0 * M_PI;

  // Amplitude such that |F(k)|^2 = A k^xi
   if (sharp){
     if(ik == kf){
        re = A / sqrt(2.) ;
	im = A / sqrt(2.) ;
        data[i].x = re;
        data[i].y = im;	
     }
   }else{
   double KF = (double) kf;
   double KK = (double) ik;
   double amplitude = gaussian(KK, KF, dk, A);

   re = amplitude * cos(phi);
   im = amplitude * sin(phi);

    data[i].x = re;
    data[i].y = im;
   }
   if ( i == 0 ) {
     data[i].x = 0.;
     data[i].y = 0.;
   }
}
//--------------------------------
void set_peak_spectrum(FFTArray1D& arr,
                                 double A, double dk,
                                 int kf, 
                                 unsigned long seed = 1234,
				 bool sharp = 0)
{ 
    if (arr.IsFourier){
      int block = 256;
      int grid = (arr.N + block - 1) / block;

      peak_spectrum_kernel<<<grid, block>>>(arr.d_complex,
                                          arr.N,
                                          A, dk, kf, seed, sharp);
      cudaDeviceSynchronize();
    }else{
      clean_exit_host("set_peak_spectrum works only in fourier space", 0);
    }
}
// ---------------------
__global__ void set_zero_kernel(cufftDoubleComplex* data, int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    data[i].x = 0.0;
    data[i].y = 0.0;
}
//----------------------------------------
void set_zero(FFTArray1D& arr){
    int N = arr.N;
    int block = 256;
    int grid = (N + block - 1) / block; 

    set_zero_kernel<<<grid, block>>>(arr.d_complex, arr.N);
    cudaDeviceSynchronize();
}
//---------------
__global__ void  complex_mult_kernel(cufftDoubleComplex* data,
				     cufftDoubleComplex z, int N){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  data[i] = cuCmul(data[i],z);  
}
//----------------------------
void  complex_mult_FFTArray(FFTArray1D& arr, cufftDoubleComplex z){
  int block = 256;
  int grid = (arr.N + block - 1) / block;
  complex_mult_kernel<<<grid, block>>>(arr.d_complex, z, arr.N);
}
//------------------------------------//
__global__ void derivk_kernel(cufftDoubleComplex* data, double hh, int N,
			      bool babs){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    if (i == 0) return;

   //
   int ik = fft_freq(i, N);
   double kk = (double) abs(ik) ;
   if (babs){
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
void derivk(FFTArray1D& arr, double hh,  bool babs){
	// calculates hh order derivative of array ff in Fourier space 
	// hh can be fractional including negative. The output is stored
	// in ff itself. If babs is set then the absolute value of the 
       // derivative is calculated. 	
    if (arr.IsFourier){
      int block = 256;
      int grid = (arr.N + block - 1) / block;
      derivk_kernel<<<grid, block>>>(arr.d_complex, hh, arr.N, babs);
    }else{
      clean_exit_host("derivk works only in fourier space", 0);
    }
}
//
__global__ void sine_kernel(cufftDoubleComplex* data, 
		 double dx, double A, int kpeak, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        double x = i * dx;
	double KPEAK = (double) kpeak;
        data[i].x = A*sin(KPEAK * x);
        data[i].y = 0.;
    }
}
void set_sine_real(cufftDoubleComplex* data, 
		 double dx, double A, int kpeak, int N){
    int block = 256;
    int grid = (N + block - 1) / block;
    sine_kernel<<<grid, block>>>(data, dx, A, kpeak, N);
}
//---------------------------------------
__global__ void cosine_kernel(cufftDoubleComplex* data, 
		   int N, double dx, double A, int kpeak) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        double x = i * dx;
	double KPEAK = (double) kpeak;
        data[i].x = A*cos(KPEAK * x);
        data[i].y = 0.;

    }
}
void set_cosine_real(cufftDoubleComplex* data, 
		 double dx, double A, int kpeak, int N){
    int block = 256;
    int grid = (N + block - 1) / block;
    cosine_kernel<<<grid, block>>>(data, N, dx, A, kpeak);
}
//-----------------------//
__global__ void exp_ix_kernel(cufftDoubleComplex* data, 
		   int N, double dx, double A, int kpeak) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        double x = i * dx;
	double KPEAK = (double) kpeak;
        data[i].x = A*cos(KPEAK * x);
        data[i].y = A*sin(KPEAK * x);
    }
}
void exp_ix(cufftDoubleComplex* data, 
		 double dx, double A, int kpeak, int N){
    int block = 256;
    int grid = (N + block - 1) / block;
    exp_ix_kernel<<<grid, block>>>(data, N, dx, A, kpeak);
}
