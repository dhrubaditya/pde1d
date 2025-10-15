#pragma once

#include <cuda_runtime.h>

// Simple macro for default block size
#ifndef RK4_BLOCK_SIZE
#define RK4_BLOCK_SIZE 256
#endif

// ---------------------------------------------------------------------
// Struct holding all device arrays needed for RK4
// ---------------------------------------------------------------------
struct TimeStepDeviceData {
    double* d_Y = nullptr;
    double* d_Ytemp  = nullptr; // device temp state
    double* d_k1     = nullptr; // dt * f(Y) stage arrays
    double* d_k2     = nullptr;
    double* d_k3     = nullptr;
    double* d_k4     = nullptr;

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
void ExpScheme(double* d_psi, int N,  double dt, 
		       TimeStepDeviceData& dev)


