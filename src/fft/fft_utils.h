#pragma once
#include <cufft.h>
#include <cuda_runtime.h>
#include <iostream>

// Structure for data buffers (no plans)
struct FFTArray1D {
    int N;                          // number of real samples
    double* d_real;                 // pointer to real data (N doubles)
    cufftDoubleComplex* d_complex;  // pointer to complex data (N/2 + 1 complex numbers)
};

// Allocate GPU memory for in-place R2C/C2R transforms
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
    cufftPlan1d(&p.plan_fwd, N, CUFFT_D2Z, 1);
    cufftPlan1d(&p.plan_inv, N, CUFFT_Z2D, 1);
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
void compute_spectrum(const FFTArray1D& arr, double* d_spectrum);
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
                                 unsigned long seed );
void set_zero(FFTArray1D& arr);
void derivk(FFTArray1D& arr, double hh,  bool abs);
void copy_FFTArray(FFTArray1D& A, FFTArray1D& B);
void cube_FFTArray(FFTArray1D& A);
void copy_FFTArray_host(double* h_A, FFTArray1D& A);
void set_sine_real(double* d_data, double dx, double A, int kpeak, int N);
void set_cosine_real(double* d_data, double dx, double A, int kpeak, int N);
void complex_mult_FFTArray(FFTArray1D& arr, cufftDoubleComplex z);
void  double2FFTArray(FFTArray1D& Arr, double* yy, int N){
  Arr.N = N;
  Arr.d_real = yy;
  Arr.d_complex = reinterpret_cast<cufftDoubleComplex*>(Arr.d_real);
}
