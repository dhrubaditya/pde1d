#pragma once

#include <curand.h>
#include <cuda_runtime.h>

// Initializes the global Mersenne Twister (MTGP32) generator with a given seed.
__host__ void rng_init(unsigned long long seed);

// Frees GPU RNG resources.
__host__ void rng_destroy();

// Generates N uniformly distributed random doubles in (0,1)
// and stores them into the device array 'd_data'.
__host__ void rng_generate_uniform(double *d_data, size_t N);

// Generates N normally distributed random doubles with given mean and stddev
// and stores them into the device array 'd_data'.
__host__ void rng_generate_normal(double *d_data, size_t N, double mean, double stddev);
// Device function to generate M uniform random numbers 
__device__ void rand_device(double *out, int M , unsigned long long seed );

