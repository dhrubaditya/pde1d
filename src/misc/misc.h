#pragma once
#include <iostream>
#include <cufft.h>

__device__ double gaussian(double x, double x_0, double sigma, double A);
__device__ cufftDoubleComplex cuCpow(cufftDoubleComplex z, double n); 

