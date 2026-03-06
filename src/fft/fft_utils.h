#pragma once
#include <cufft.h>
#include <cuda_runtime.h>
#include <iostream>

// Structure for data buffers (no plans)
struct FFTArray1D {
    int N;                          // number of real samples
    cufftDoubleComplex* d_complex = nullptr;// pointer to complex data
    bool IsFourier = true;
};

// Allocate GPU memory for in-place C2C transforms
FFTArray1D fft_alloc_1d(int N);
// Free memory
void fft_free_1d(FFTArray1D& arr);


// Separate plan management
struct FFTPlan1D {
    cufftHandle plan_fwd;
    cufftHandle plan_inv;
    int N;
};

// Create reusable FFT plans (for any arrays of same N)
inline FFTPlan1D fft_plan_create_1d(int N) {
    FFTPlan1D p;
    p.N = N;
    cufftPlan1d(&p.plan_fwd, N, CUFFT_Z2Z, 1);
    cufftPlan1d(&p.plan_inv, N, CUFFT_Z2Z, 1);
    return p;
}

// Destroy reusable FFT plans
inline void fft_plan_destroy_1d(FFTPlan1D& p) {
    cufftDestroy(p.plan_fwd);
    cufftDestroy(p.plan_inv);
}

void fft_forward_inplace(const FFTPlan1D& plan, FFTArray1D& arr);
void fft_inverse_inplace(const FFTPlan1D& plan, FFTArray1D& arr);
// Normalize real-space array after inverse FFT
void normalize_fft(FFTArray1D& arr);
// -------------------------------
// Spectrum calculation
// -------------------------------

// Compute the power spectrum |F(k)|^2 of a real-to-complex FFT array
// d_spectrum should be a device pointer of size N/2 + 1
void compute_normalized_spectrum(double* d_spectrum, const FFTArray1D& arr);
// Optional: normalize a spectrum array (|F(k)|^2)
void normalize_spectrum(double* d_spectrum, int N);
//-----------------
void set_power_law_spectrum(FFTArray1D& arr,
                                      double A, double xi,
                                      int kmin, int kmax,
                                      unsigned long seed);
void set_peak_spectrum(FFTArray1D& arr,
                                 double A, double dk,
                                 int kf,
                                 unsigned long seed,
	                         bool sharp);
void set_ksqr_exp_k_spectrum(FFTArray1D& arr,
                                 double A, double dk,
                                 int kf,
                                 unsigned long seed);
void set_white_spectrum(FFTArray1D& arr,
                                 double A, double dk,
			         int kcut,
                                 unsigned long seed);
void test_fft_freq(int N);
void set_zero(FFTArray1D& arr);
void set_zero_cmplx_array(cufftDoubleComplex* A, int N);
void derivk(FFTArray1D& arr, double hh,  bool abs);
void copy_FFTArray(FFTArray1D& B, const FFTArray1D& A);
//void copy_FFTArray_host(double* h_A, FFTArray1D& A);
void AssociateComplex2FFT(FFTArray1D& AFFT, cufftDoubleComplex* d_A, 
                int N, bool IsF);
void copy_FFTArray_host_complex(cufftDoubleComplex* h_arr, FFTArray1D& arr);
void set_sine_real(double* d_data, double dx, double A, int kpeak, int N);
void set_cosine_real(double* d_data, double dx, double A, int kpeak, int N);
void exp_ix(cufftDoubleComplex* data,
                 double dx, double A, int kpeak, int N);
void complex_mult_FFTArray(FFTArray1D& arr, cufftDoubleComplex z);
void  AtimesX(cufftDoubleComplex* A, double X, int N);
//void double2FFTArray(FFTArray1D& Arr, double* yy, int N);
void complex2FFTArray(FFTArray1D& Arr, 
		     const cufftDoubleComplex* yy, int N, bool is_fourier);
void abs2_times_z_FFTArray(FFTArray1D& A);
__device__ __host__ int fft_freq(int i, int N);
