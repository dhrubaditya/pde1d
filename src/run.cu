#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cstdio>
#include "start.h"
#include "run.h"
#include "model.h"
#include "io.h"
#include "fft_utils.h"
#include "initcond.h"
#include "misc.h"
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
  // section 1 : Start  Parameters
    const SParams h_Sparams = read_Sparams("./input/start.in");

    std::cout << "Simulation parameters:" << std::endl;
    std::cout << "NX = " << h_Sparams.NX << std::endl;
    std::cout << "DX = " << h_Sparams.DX << std::endl;
    std::cout << "DK = " << h_Sparams.DK << std::endl;

// endsection
// section 2 : read initial condition 
    int N = h_Sparams.NX;
    double dk = h_Sparams.DK;
    double dx = h_Sparams.DX;

    // --- Host memory  ---
    cufftDoubleReal *psi, *psik;
    CUDA_CHECK(cudaMallocHost((void**)&psi,
			      sizeof(cufftDoubleReal) * (N + 2) ));
    CUDA_CHECK(cudaMallocHost((void**)&psik,
                               sizeof(cufftDoubleReal) * (N + 2) ));
    // N+2 because fft needs extra storage.
    cufftDoubleReal *Ek;
    CUDA_CHECK(cudaMallocHost((void**)&Ek,
    			      sizeof(double) * (N/2 + 1) ));
    // device memory
    FFTArray1D d_psi = fft_alloc_1d(N);
    //
    std::cout << "Reading run parameters .." << std::endl;
    const RParams h_Rparams = read_Rparams("./input/run.in");
    std::cout << "..done" << std::endl;
    if(h_Rparams.run){
      read_initcond(psi, psik, N, RParams);
    }else{
      clean_exit_host("No initial condition, run start first",1);
    }
 // section 3 : Model
    setup_model(N);
 //  section 4 : timestepping
    TimeStepDeviceData TStep = TimeStep_allocate_device_memory(N + 2);
    // N + 2 because of FFT structure
    DiagData Diag = setup_diag(N);
    double time = 0.
    for (int iouter = 0; iouter < h_Rparams.NITER/h_Rparams.NAVG; iouter++){
      for(int iinner = 0; iinner < h_Rparams.NAVG; iinner++){
	ExpScheme(d_psi, N + 2, RParams.dt, TStep);
	time = time + dt;
      }
      compute_diag(d_psi, Diag);
    }
    write_diag(Diag);
    write_data(d_psi);
// endsection

// section : clean up 
    cudaFreeHost(psi); cudaFreeHost(psik);
    fft_free_1d(d_psi);
    TimeStep_free_device_memory(dev);
    cleanup_model();
    std::cout << "Code Exited Cleanly" << std::endl;
//
    return 0;
}

