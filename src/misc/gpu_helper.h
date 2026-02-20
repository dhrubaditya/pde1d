#pragma once 

#include <cuda_runtime.h>
#include <stdio.h>

// Queries the properties of the first available CUDA device and stores them
// in the provided cudaDeviceProp struct.
// Returns cudaSuccess on success, or a cudaError_t code on failure.
cudaError_t getGpuProperties(cudaDeviceProp *prop, int deviceId);

// Writes the GPU properties from the provided cudaDeviceProp struct to a file.
// Returns 0 on success, -1 on failure (e.g. file could not be opened).
int writeGpuPropertiesToFile(const cudaDeviceProp *prop, int deviceId, const char *filename);

