#include "random.h"
#include <iostream>

static curandGenerator_t gen;
static bool initialized = false;

__host__ void rng_init(unsigned long long seed)
{
    if (initialized) return;

    curandStatus_t status = curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MTGP32);
    if (status != CURAND_STATUS_SUCCESS) {
        std::cerr << "Error: Failed to create cuRAND MTGP32 generator\n";
        return;
    }

    status = curandSetPseudoRandomGeneratorSeed(gen, seed);
    if (status != CURAND_STATUS_SUCCESS) {
        std::cerr << "Error: Failed to set cuRAND seed\n";
        return;
    }

    initialized = true;
}

__host__ void rng_generate_uniform(double *d_data, size_t N)
{
    if (!initialized) {
        std::cerr << "Error: RNG not initialized. Call rng_init() first.\n";
        return;
    }

    curandStatus_t status = curandGenerateUniformDouble(gen, d_data, N);
    if (status != CURAND_STATUS_SUCCESS) {
        std::cerr << "Error: cuRAND failed to generate uniform doubles\n";
    }
}

__host__ void rng_generate_normal(double *d_data, size_t N, double mean, double stddev)
{
    if (!initialized) {
        std::cerr << "Error: RNG not initialized. Call rng_init() first.\n";
        return;
    }

    curandStatus_t status = curandGenerateNormalDouble(gen, d_data, N, mean, stddev);
    if (status != CURAND_STATUS_SUCCESS) {
        std::cerr << "Error: cuRAND failed to generate normal doubles\n";
    }
}

__host__ void rng_destroy()
{
    if (!initialized) return;
    curandDestroyGenerator(gen);
    initialized = false;
}

