#pragma once

#include <cuda_runtime.h>
#include <cufft.h>
#include <iostream>
// Simple macro for default block size
#ifndef RK4_BLOCK_SIZE
#define RK4_BLOCK_SIZE 256
#endif

// ---------------------------------------------------------------------
// Struct holding all device arrays needed for RK4
// ---------------------------------------------------------------------
struct TimeStepDeviceData {
    cufftDoubleComplex* d_Y = nullptr;
    cufftDoubleComplex* d_Ytemp  = nullptr; // device temp state
    cufftDoubleComplex* d_k1     = nullptr; // dt * f(Y) stage arrays
    cufftDoubleComplex* d_k2     = nullptr;
    cufftDoubleComplex* d_k3     = nullptr;
    cufftDoubleComplex* d_k4     = nullptr;

    // Host-side flag: 
    bool is_initialized = false;
};

// ---------------------------------------------------------------------
// Allocate all device memory needed for RK4
//   N : size of the system (length of Y)
// ---------------------------------------------------------------------
TimeStepDeviceData TimeStep_allocate_device_memory(int N);
// ---------------------------------------------------------------------
// Free all previously allocated device memory
// ---------------------------------------------------------------------
void TimeStep_free_device_memory(TimeStepDeviceData& dev);
// ---------------------------------------------------------------------
void ExpScheme(cufftDoubleComplex* d_psi, int N,  double dt, 
		       TimeStepDeviceData& dev);
