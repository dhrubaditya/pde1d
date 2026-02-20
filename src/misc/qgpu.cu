#include "gpu_helper.h"

int main() {
    int deviceId = 0;
    cudaDeviceProp prop;

    if (getGpuProperties(&prop, deviceId) != cudaSuccess) {
        return -1;
    }

    if (writeGpuPropertiesToFile(&prop, deviceId, "gpu_properties.txt") != 0) {
        return -1;
    }

}
