#pragma once
#include <cufft.h>
#include <cuda_runtime.h>
#include <iostream>
#include "fft_utils.h"
void setup_model(int N);
void cleanup_model();
void test_model_param();
// ------------------------------------------------------
// Compute RHS 
// ------------------------------------------------------
void compute_rhs(cufftDoubleComplex* RHS, const cufftDoubleComplex* d_psik, 
		 double tt, int N, int stage);
void exp_transform(cufftDoubleComplex* psik,
                   cufftDoubleComplex* vv, double time,
                   bool mult_by_i, int N );
cufftDoubleComplex test_NN_conservation(FFTArray1D& psik);
void copy_NLIN2host(cufftDoubleComplex* h_nlin, double* h_nlink,
                    const FFTArray1D& d_psi);
