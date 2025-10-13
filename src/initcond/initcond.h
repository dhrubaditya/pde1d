#pragma once
#include <cufft.h>
#include <cuda_runtime.h>
#include <string>
#include "fft_utils.h"
struct IParams {
    bool FOURIER = true;
    std::string ITYPE = "zero";
    double A = 0.0;
    double xi = 0.0;
    int kmax = 0;
    int kmin = 0;
    int  kpeak = 1;
};
void set_initcond(FFTArray1D& d_arr, double dk, double dx, IParams IC);

