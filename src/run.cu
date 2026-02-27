#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cstdio>
#include <iomanip>
#include "start.h"
#include "run.h"
#include "model.h"
#include "io.h"
#include "fft_utils.h"
#include "initcond.h"
#include "misc.h"
#include "gpu_helper.h"
#include "evolve.h"

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
    // section 1 : Start  Parameters
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
    cufftDoubleComplex *psi;
    cufftDoubleComplex *psik;
    CUDA_CHECK(cudaMallocHost((void**)&psi,
                              sizeof(cufftDoubleComplex) * N  ));
    CUDA_CHECK(cudaMallocHost((void**)&psik,
                               sizeof(cufftDoubleComplex) * N ));
    // Remember we are arrays that are complex in real space.   
    cufftDoubleReal *Ek;
    CUDA_CHECK(cudaMallocHost((void**)&Ek,
                              sizeof(double) * (N/2 + 1)  ));
    // device memory
    FFTArray1D d_psi = fft_alloc_1d(N);
    FFTPlan1D plan = fft_plan_create_1d(N);
    double* d_Ek;
    CUDA_CHECK(cudaMalloc(&d_Ek, sizeof(double) * (N/2 + 1) ) );
// setup up initial condition (will be responsibility
// of start.cu later. 
    std::cout << "Reading initial condition input/icond.in .." << std::endl;
    const IParams h_Iparams = read_icond("./input/icond.in");
    std::cout << "..done" << std::endl;
    std::cout << "Generating initial condition (in device) .." << std::endl;
    if (h_Iparams.FOURIER){
      set_initcond(d_psi, dk, dx, h_Iparams);
      compute_normalized_spectrum(d_Ek, d_psi);
      copy_FFTArray_host_complex(psik, d_psi);
    }else{
      clean_exit_host("run: initial condition in real not coded", 0);
    }
    CUDA_CHECK(cudaMemcpy(Ek, d_Ek, sizeof(double) * (N/2 + 1),
                          cudaMemcpyDeviceToHost));
    std::cout << "Writing intial condition to files .." << std::endl;
    write_complex_array(psik, dk, N, "inicond.out");
    std::cout << "..done" << std::endl;
    std::cout << "Writing intial energy spectrum file .." << std::endl;
    write_spectrum(Ek, N, dk, 0);
    std::cout << "..done" << std::endl;
    //
    std::cout << "Reading run parameters .." << std::endl;
    const RParams h_Rparams = read_Rparams("./input/run.in");
    std::cout << "dt = " << h_Rparams.dt << std::endl;
    std::cout << "..done" << std::endl; 
    double dt = h_Rparams.dt;
// section 2 : read initial condition 
/*
    if(h_Rparams.run){
      read_initcond(psi, psik, N, RParams);
    }else{
      clean_exit_host("No initial condition, run start first",1);
    } */
 // section 3 : Model
    setup_model(N);
 //  section 4 : timestepping
    TimeStepDeviceData TStep = TimeStep_allocate_device_memory(N);
    // N because of complex->complex fft 
    //DiagData Diag = setup_diag(N);
    std::ofstream out("data/diag.dat");
    if (!out) {
        std::cerr << "Error: cannot open diag.dat for writing\n";
        return -1;
    }
    out << std::scientific << std::setprecision(8);

    double time = 0.;
    std::cout << "starting timestepping, time=:\t"<< time << std::endl;
    for (int iouter = 0; iouter < h_Rparams.NITER/h_Rparams.NAVG; iouter++){
      for(int iinner = 0; iinner < h_Rparams.NAVG; iinner++){
	ExpScheme(d_psi.d_complex, N, dt, TStep);
	time = time + dt;
      }
      std::cout << "running, time=:\t"<< time << std::endl;
      std::cout << "computing spectrum and writing to file .." << std::endl;
      calc_diag(d_Ek, d_psi, time, out);
      CUDA_CHECK(cudaMemcpy(Ek, d_Ek, sizeof(double) * (N/2 + 1),
                          cudaMemcpyDeviceToHost));
      write_spectrum(Ek, N, dk, iouter+1);
      std::cout << "..done" << std::endl;
    }
    //write_diag(Diag);
    //write_data(d_psi);
    out.close();
// endsection
// section : clean up 
//  First check how much memory has been used
    size_t usedBytes, freeBytes, totalBytes;
    if (getDeviceMemoryUsage(deviceId, &usedBytes, &freeBytes, &totalBytes)
                    == cudaSuccess) {
        printf("Device %d memory usage:\n", deviceId);
        printf("  Used  : %.2f MB\n", (double)usedBytes  / (1024.0 * 1024.0));
        printf("  Free  : %.2f MB\n", (double)freeBytes  / (1024.0 * 1024.0));
        printf("  Total : %.2f MB\n", (double)totalBytes / (1024.0 * 1024.0));
    }

    cudaFreeHost(psi); cudaFreeHost(psik);
    fft_free_1d(d_psi);
    TimeStep_free_device_memory(TStep);
    cleanup_model();
    std::cout << "Code Exited Cleanly" << std::endl;
//
    return 0;
}
