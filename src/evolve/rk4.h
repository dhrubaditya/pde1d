#ifndef RK4_H
#define RK4_H

#include <cuda_runtime.h>

// Simple macro for default block size
#ifndef RK4_BLOCK_SIZE
#define RK4_BLOCK_SIZE 256
#endif

// ---------------------------------------------------------------------
// Struct holding all device arrays needed for RK4
// ---------------------------------------------------------------------
struct RK4DeviceData {
    double* d_Y      = nullptr; // device state vector
    double* d_Ytemp  = nullptr; // device temp state
    double* d_k1     = nullptr; // dt * f(Y) stage arrays
    double* d_k2     = nullptr;
    double* d_k3     = nullptr;
    double* d_k4     = nullptr;

    // Host-side flag: false until first copy host->device performed.
    bool is_initialized = false;
};

// ---------------------------------------------------------------------
// Allocate all device memory needed for RK4
//   N : size of the system (length of Y)
// Returns an RK4DeviceData struct with allocated device pointers
// ---------------------------------------------------------------------
RK4DeviceData rk4_allocate_device_memory(int N);

// ---------------------------------------------------------------------
// Free all previously allocated device memory
// ---------------------------------------------------------------------
void rk4_free_device_memory(RK4DeviceData& dev);

// ---------------------------------------------------------------------
// Perform M RK4 timesteps on a system of size N
//   Y_host : input/output state on the host
//   N      : size of state vector (number of doubles)
//   dt     : timestep
//   M      : number of RK4 steps to perform (can be 0 to only init device copy)
//   dev    : struct with pre-allocated device memory (see allocator above)
//
// Behavior:
//  - On the first call for a given RK4DeviceData (dev.is_initialized==false),
//    the function copies Y_host -> device and sets is_initialized = true.
//  - Subsequent calls will NOT copy host->device (device keeps authoritative state).
//  - At the end of the call, it copies the updated device state back to Y_host.
//    If you prefer to skip that copy, you can modify the implementation.
// ---------------------------------------------------------------------
void rk4_timestep_host(double* Y_host, int N, double dt, int M, RK4DeviceData& dev);

#endif // RK4_H

