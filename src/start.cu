#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cstdio>
#include "start.h"
#include "io.h"
#include "fft_utils.h"
#include "initcond.h"
#include "misc.h"
#include "gpu_helper.h"
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error %s:%d: '%s'\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                               \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)
// ******************************************** //
int main() {

    // section 0 : query to GPU 
    std::cout << "Checking GPU properties" << std::endl;  
    int deviceId = 0;
    cudaDeviceProp prop;

    if (getGpuProperties(&prop, deviceId) != cudaSuccess) {
        return -1;
    }

    if (writeGpuPropertiesToFile(&prop, deviceId, "gpu_prop.txt") != 0) {
        return -1;
    }
    std::cout << "GPU properties written to file <gpu_prop.txt>" << std::endl;  
    // section 1 : Input  Parameters
    const SParams h_Sparams = read_Sparams("./input/start.in");

    std::cout << "Simulation parameters:" << std::endl;
    std::cout << "NX = " << h_Sparams.NX << std::endl;
    std::cout << "DX = " << h_Sparams.DX << std::endl;
    std::cout << "DK = " << h_Sparams.DK << std::endl;

// endsection
// section 2 : initial condition 
    int N = h_Sparams.NX;
    double dk = h_Sparams.DK;
    double dx = h_Sparams.DX;

    // --- Host memory  ---
    cufftDoubleComplex *psi, *psik;
    CUDA_CHECK(cudaMallocHost((void**)&psi,
			      sizeof(cufftDoubleComplex) * N  ));
    CUDA_CHECK(cudaMallocHost((void**)&psik,
                               sizeof(cufftDoubleComplex) * N  ));
    // complex to complex fft no extra storage 
    cufftDoubleReal *Ek;
    CUDA_CHECK(cudaMallocHost((void**)&Ek,
    			      sizeof(double) * N ));
    // device memory
    FFTArray1D d_psi = fft_alloc_1d(N);
    FFTPlan1D plan = fft_plan_create_1d(N);
    double* d_Ek;
    CUDA_CHECK(cudaMalloc(&d_Ek, sizeof(double) * N ) );

    std::cout << "Reading initial condition input/icond.in .." << std::endl;
    const IParams h_Iparams = read_icond("./input/icond.in");
    std::cout << "..done" << std::endl;
    std::cout << "Generating initial condition (in device) .." << std::endl;
    if (h_Iparams.FOURIER){
      set_initcond(d_psi, dk, dx, h_Iparams);
      compute_normalized_spectrum(d_Ek, d_psi);
      copy_FFTArray_host_complex(psik, d_psi);
      fft_inverse_inplace(plan, d_psi);
      normalize_fft(d_psi);
      copy_FFTArray_host_complex(psi, d_psi);
    }else{
      set_initcond(d_psi, dk, dx, h_Iparams);
      copy_FFTArray_host_complex(psi, d_psi);
      fft_forward_inplace(plan, d_psi);
      compute_normalized_spectrum(d_Ek, d_psi);
      copy_FFTArray_host_complex(psik, d_psi);
    }
    CUDA_CHECK(cudaMemcpy(Ek, d_Ek, sizeof(double) * N,
			  cudaMemcpyDeviceToHost));
    std::cout << "Writing intial condition to files .." << std::endl;
    write_psi(psi, psik, "initcond", dx, dk, N);
    std::cout << "..done" << std::endl;
    std::cout << "Writing intial energy spectrum file .." << std::endl;
    write_spectrum(Ek, N, dk, 0);
    std::cout << "..done" << std::endl;
// endsection
//    std::cout << "Setup models .." << std::endl;
//    setup_model(N); 
//    std::cout << "Testing conservation of NN .." << std::endl;
//    cufftDoubleComplex RHSN = test_NN_conservation(d_psi);
//    std::cout << "RHS is .." << std::endl;
//    std::cout << RHSN.x << " " << RHSN.y << std::endl;
// section : clean up
//  First check how memory has been used
    size_t usedBytes, freeBytes, totalBytes;
    if (getDeviceMemoryUsage(deviceId, &usedBytes, &freeBytes, &totalBytes) 
		    == cudaSuccess) {
        printf("Device %d memory usage:\n", deviceId);
        printf("  Used  : %.2f MB\n", (double)usedBytes  / (1024.0 * 1024.0));
        printf("  Free  : %.2f MB\n", (double)freeBytes  / (1024.0 * 1024.0));
        printf("  Total : %.2f MB\n", (double)totalBytes / (1024.0 * 1024.0));
    }
// Now free memory
    cudaFreeHost(psi); cudaFreeHost(psik); cudaFreeHost(Ek);
    fft_plan_destroy_1d(plan);
    fft_free_1d(d_psi);
    cudaFree(d_Ek);
//
    return 0;
}

