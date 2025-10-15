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
void compute_rhs(const double* d_vv, const double* d_psik, double* RHS,
		 double tt, double dt, int N,
		 int stage);
void exp_transform(double* psik, double* vv, double time, int N );
void exp_inv_transform(double* vv, double* psik, double time, int N );

