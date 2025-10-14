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
void compute_rhs(FFTPlan1D& plan, FFTArray1D& Y, FFTArray1D& RHS, double dt);
