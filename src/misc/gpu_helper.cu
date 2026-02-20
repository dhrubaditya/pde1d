#include "gpu_helper.h"

cudaError_t getGpuProperties(cudaDeviceProp *prop, int deviceId) {
    if (prop == NULL) {
        fprintf(stderr, "[getGpuProperties] Error: NULL pointer provided for prop.\n");
        return cudaErrorInvalidValue;
    }

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "[getGpuProperties] Error getting device count: %s\n",
                cudaGetErrorString(err));
        return err;
    }

    if (deviceCount == 0) {
        fprintf(stderr, "[getGpuProperties] Error: No CUDA-capable devices found.\n");
        return cudaErrorNoDevice;
    }

    if (deviceId < 0 || deviceId >= deviceCount) {
        fprintf(stderr, "[getGpuProperties] Error: deviceId %d is out of range (0-%d).\n",
                deviceId, deviceCount - 1);
        return cudaErrorInvalidDevice;
    }

    err = cudaGetDeviceProperties(prop, deviceId);
    if (err != cudaSuccess) {
        fprintf(stderr, "[getGpuProperties] Error getting device properties: %s\n",
                cudaGetErrorString(err));
        return err;
    }

    return cudaSuccess;
}

int writeGpuPropertiesToFile(const cudaDeviceProp *prop, int deviceId, const char *filename) {
    if (prop == NULL || filename == NULL) {
        fprintf(stderr, "[writeGpuPropertiesToFile] Error: NULL pointer provided.\n");
        return -1;
    }

    FILE *f = fopen(filename, "w");
    if (f == NULL) {
        fprintf(stderr, "[writeGpuPropertiesToFile] Error: Could not open file '%s'.\n", filename);
        return -1;
    }

    fprintf(f, "============================================================\n");
    fprintf(f, "  GPU Device Properties  (Device ID: %d)\n", deviceId);
    fprintf(f, "============================================================\n");

    // --- Identity ---
    fprintf(f, "\n[Identity]\n");
    fprintf(f, "  Device Name                    : %s\n",   prop->name);
    fprintf(f, "  Compute Capability             : %d.%d\n", prop->major, prop->minor);
    fprintf(f, "  PCI Bus ID                     : %d\n",   prop->pciBusID);
    fprintf(f, "  PCI Device ID                  : %d\n",   prop->pciDeviceID);
    fprintf(f, "  PCI Domain ID                  : %d\n",   prop->pciDomainID);

    // --- Memory ---
    fprintf(f, "\n[Memory]\n");
    fprintf(f, "  Total Global Memory            : %.2f MB\n",
            (double)prop->totalGlobalMem / (1024.0 * 1024.0));
    fprintf(f, "  Total Constant Memory          : %.2f KB\n",
            (double)prop->totalConstMem / 1024.0);
    fprintf(f, "  Shared Memory per Block        : %.2f KB\n",
            (double)prop->sharedMemPerBlock / 1024.0);
    fprintf(f, "  Shared Memory per SM           : %.2f KB\n",
            (double)prop->sharedMemPerMultiprocessor / 1024.0);
    fprintf(f, "  L2 Cache Size                  : %.2f KB\n",
            (double)prop->l2CacheSize / 1024.0);
    fprintf(f, "  Memory Bus Width               : %d bits\n", prop->memoryBusWidth);
    fprintf(f, "  Memory Clock Rate              : %.2f MHz\n",
            prop->memoryClockRate / 1000.0);
    fprintf(f, "  ECC Enabled                    : %s\n",
            prop->ECCEnabled ? "Yes" : "No");

    // --- Compute ---
    fprintf(f, "\n[Compute]\n");
    fprintf(f, "  Number of SMs                  : %d\n",   prop->multiProcessorCount);
    fprintf(f, "  Clock Rate                     : %.2f MHz\n",
            prop->clockRate / 1000.0);
    fprintf(f, "  Max Threads per Block          : %d\n",   prop->maxThreadsPerBlock);
    fprintf(f, "  Max Threads per SM             : %d\n",   prop->maxThreadsPerMultiProcessor);
    fprintf(f, "  Warp Size                      : %d\n",   prop->warpSize);
    fprintf(f, "  Registers per Block            : %d\n",   prop->regsPerBlock);
    fprintf(f, "  Registers per SM               : %d\n",   prop->regsPerMultiprocessor);
    fprintf(f, "  Max Block Dimensions           : (%d, %d, %d)\n",
            prop->maxThreadsDim, prop->maxThreadsDim, prop->maxThreadsDim);
    fprintf(f, "  Max Grid Dimensions            : (%d, %d, %d)\n",
            prop->maxGridSize, prop->maxGridSize, prop->maxGridSize);

    // --- Features ---
    fprintf(f, "\n[Features]\n");
    fprintf(f, "  Unified Addressing             : %s\n",
            prop->unifiedAddressing ? "Yes" : "No");
    fprintf(f, "  Managed Memory                 : %s\n",
            prop->managedMemory ? "Yes" : "No");
    fprintf(f, "  Concurrent Kernels             : %s\n",
            prop->concurrentKernels ? "Yes" : "No");
    fprintf(f, "  Async Engine Count             : %d\n",   prop->asyncEngineCount);
    fprintf(f, "  Can Map Host Memory            : %s\n",
            prop->canMapHostMemory ? "Yes" : "No");
    fprintf(f, "  Cooperative Launch             : %s\n",
            prop->cooperativeLaunch ? "Yes" : "No");
    fprintf(f, "  Compute Mode                   : %d\n",   prop->computeMode);
    fprintf(f, "  TCC Driver                     : %s\n",
            prop->tccDriver ? "Yes" : "No");

    fprintf(f, "\n============================================================\n");

    fclose(f);
    return 0;
}
//----------------------------------------//
cudaError_t getDeviceMemoryUsage(int deviceId, size_t *usedBytes, size_t *freeBytes, size_t *totalBytes) {

    // At least one output pointer must be provided
    if (usedBytes == NULL && freeBytes == NULL && totalBytes == NULL) {
        fprintf(stderr, "[getDeviceMemoryUsage] Error: All output pointers are NULL.\n");
        return cudaErrorInvalidValue;
    }

    // Save the currently active device so we can restore it afterwards
    int previousDeviceId = 0;
    cudaError_t err = cudaGetDevice(&previousDeviceId);
    if (err != cudaSuccess) {
        fprintf(stderr, "[getDeviceMemoryUsage] Error getting current device: %s\n",
                cudaGetErrorString(err));
        return err;
    }

    // Switch to the requested device if necessary
    if (deviceId != previousDeviceId) {
        err = cudaSetDevice(deviceId);
        if (err != cudaSuccess) {
            fprintf(stderr, "[getDeviceMemoryUsage] Error setting device %d: %s\n",
                    deviceId, cudaGetErrorString(err));
            return err;
        }
    }

    // Query free and total memory on the active device
    size_t localFree  = 0;
    size_t localTotal = 0;
    err = cudaMemGetInfo(&localFree, &localTotal);
    if (err != cudaSuccess) {
        fprintf(stderr, "[getDeviceMemoryUsage] Error querying memory info: %s\n",
                cudaGetErrorString(err));
        // Restore the original device before returning
        if (deviceId != previousDeviceId) {
            cudaSetDevice(previousDeviceId);
        }
        return err;
    }

    // Restore the original device if we switched
    if (deviceId != previousDeviceId) {
        err = cudaSetDevice(previousDeviceId);
        if (err != cudaSuccess) {
            fprintf(stderr, "[getDeviceMemoryUsage] Error restoring device %d: %s\n",
                    previousDeviceId, cudaGetErrorString(err));
            return err;
        }
    }

    // Write results to the provided output pointers
    if (freeBytes  != NULL) *freeBytes  = localFree;
    if (totalBytes != NULL) *totalBytes = localTotal;
    if (usedBytes  != NULL) *usedBytes  = localTotal - localFree;

    return cudaSuccess;
}
