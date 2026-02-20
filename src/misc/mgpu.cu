#include "gpu_helper.h"

int main() {
    int deviceId = 0;

    // ... allocations and kernel launches ...

    // --- end of code ---
    size_t usedBytes, freeBytes, totalBytes;
    if (getDeviceMemoryUsage(deviceId, &usedBytes, &freeBytes, &totalBytes) == cudaSuccess) {
        printf("Device %d memory usage:\n", deviceId);
        printf("  Used  : %.2f MB\n", (double)usedBytes  / (1024.0 * 1024.0));
        printf("  Free  : %.2f MB\n", (double)freeBytes  / (1024.0 * 1024.0));
        printf("  Total : %.2f MB\n", (double)totalBytes / (1024.0 * 1024.0));
    }

    return 0;
}
