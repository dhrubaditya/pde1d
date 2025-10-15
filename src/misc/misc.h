#pragma once
#include <iostream>
#include <cufft.h>

struct GpuReducer {
    double* d_partial = nullptr;
    double* h_partial = nullptr;
    size_t max_size = 0;
};

__device__ double gaussian(double x, double x_0, double sigma, double A);
__device__ cufftDoubleComplex cuCpow(cufftDoubleComplex z, double n);
void clean_exit_host(const std::string &msg = "",
			      bool mark_incomplete = false);
void init_reducer(GpuReducer &red, size_t N);
double gpu_sum(const double* d_Arr, size_t N, GpuReducer &red);
void free_reducer(GpuReducer &red);
__host__ __device__  cufftDoubleComplex exp_cuComplex(cufftDoubleComplex G,
						      double t);
