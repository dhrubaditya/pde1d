#include <iostream>
#include <fstream>
#include <cmath>
#include <cuComplex.h>
#include <math.h>
#include <cuda_runtime.h>
#include "misc.h"

__device__ double gaussian(double x, double x_0, double sigma, double A) 
{
    // Gaussian centered at x_0, width parameter sigma, amplitude A
    double arg = (x - x_0) / (sigma);
    return A * exp(-arg * arg);
}
//
__device__ cufftDoubleComplex cuCpow(cufftDoubleComplex z, double n) {
    double r = cuCabs(z);          // magnitude = sqrt(x^2 + y^2)
    double theta = atan2(z.y, z.x); // argument = phase
    double rn = pow(r, n);         // magnitude^n
    double ang = n * theta;        // n * phase

    return make_cuDoubleComplex(rn * cos(ang), rn * sin(ang));
}


