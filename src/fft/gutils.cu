#include <cstdio>
#include <cuda_runtime.h>


/* ------------------------------------------------------------------
   Helper to translate (major,minor) -> CUDA cores per SM.
   Values are taken from NVIDIA's official tables (as of CUDA 12.x).
   ------------------------------------------------------------------ */
static int coresPerSM(int major, int minor)
{
    typedef struct { int major, minor, cores; } SMtoCores;
    const SMtoCores table[] = {
        { 1, 0,  8 }, { 1, 1,  8 },
        { 2, 0, 32 }, { 2, 1, 48 },
        { 3, 0, 192}, { 3, 2, 192},
        { 5, 0, 128}, { 5, 2, 128},
        { 6, 0,  64}, { 6, 1, 128},
        { 7, 0,  64}, { 7, 5,  64},
        { 8, 0,  64}, { 8, 6, 128},
        { 9, 0, 128},                 // Hopper
        { -1, -1, -1 }                // sentinel
    };

    for (int i = 0; table[i].major != -1; ++i) {
        if (table[i].major == major && table[i].minor == minor)
            return table[i].cores;
    }
    // Unknown architecture – guess 64 cores per SM (common for newer GPUs)
    return 64;
}


int main(int argc, char *argv[])
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceCount failed: %s\n",
                cudaGetErrorString(err));
        return 1;
    }

    printf("Found %d CUDA device%s\n", deviceCount,
           deviceCount == 1 ? "" : "s");

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, dev);
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaGetDeviceProperties(%d) failed: %s\n",
                    dev, cudaGetErrorString(err));
            continue;
        }

        printf("\n=== Device %d ===\n", dev);
        printf("Name                     : %s\n",          prop.name);
        printf("Compute capability       : %d.%d\n",      prop.major, prop.minor);
        printf("Total global memory      : %llu MB\n",
               static_cast<unsigned long long>(prop.totalGlobalMem) >> 20);
        printf("Multiprocessors (SMs)    : %d\n",          prop.multiProcessorCount);
        printf("CUDA cores (approx.)     : %d\n",
               prop.multiProcessorCount *
               coresPerSM(prop.major, prop.minor));   // helper defined below
        printf("Clock rate               : %.2f MHz\n",
               prop.clockRate * 1e-3f);
        printf("Memory clock rate        : %.2f MHz\n",
               prop.memoryClockRate * 1e-3f);
        printf("Memory bus width         : %d bits\n",    prop.memoryBusWidth);
        printf("Peak memory bandwidth    : %.2f GB/s\n",
               2.0f * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1e6f);
        printf("Shared memory per block  : %zu KB\n",
               prop.sharedMemPerBlock >> 10);
        printf("Registers per block      : %d\n",          prop.regsPerBlock);
        printf("Max threads per block    : %d\n",          prop.maxThreadsPerBlock);
        printf("Max threads dimensions   : (%d, %d, %d)\n",
               prop.maxThreadsDim[0],
               prop.maxThreadsDim[1],
               prop.maxThreadsDim[2]);
        printf("Max grid size            : (%d, %d, %d)\n",
               prop.maxGridSize[0],
               prop.maxGridSize[1],
               prop.maxGridSize[2]);
        printf("Warp size                : %d\n",          prop.warpSize);
        printf("Concurrent kernels?      : %s\n",
               prop.concurrentKernels ? "yes" : "no");
        printf("ECC enabled?             : %s\n",
               prop.ECCEnabled ? "yes" : "no");
        printf("Unified addressing?      : %s\n",
               prop.unifiedAddressing ? "yes" : "no");
    }
    return 0;
}
