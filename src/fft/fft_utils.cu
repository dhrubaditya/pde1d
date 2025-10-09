#include <curand_kernel.h>
#include "fft_utils.h"
#include "misc.h"

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
//----------------------------------------
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
