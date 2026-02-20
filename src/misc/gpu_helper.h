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
// Queries the current free and total device memory and reports how much
// memory has been allocated (i.e. total - free) on the given device.
// Stores the results in the provided pointers (all values in bytes).
// Pass NULL for any output you don't need.
// Returns cudaSuccess on success, or a cudaError_t code on failure.
cudaError_t getDeviceMemoryUsage(int deviceId, size_t *usedBytes, size_t *freeBytes, size_t *totalBytes);
