#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cstdio>
#include <iomanip>
#include "model.h"
#include "fft_utils.h"
#include "misc.h"
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
int main(int argc, char** argv){
    int N = 32;              // grid size 
    double L = 2 * M_PI;       // domain size
    if (argc > 1) {
      N = std::atoi(argv[1]);
      if (N <= 0) {
        printf("Error: N must be a positive integer.\n");
        return 1;
       }
     }
     printf("Grid size N = %d\n", N);
     // section 3 : Model
     setup_model(N);
     test_model_param();
     // Get the Green's function.   
     cufftDoubleComplex *Gfunc;
     CUDA_CHECK(cudaMallocHost((void**)&Gfunc,
                              sizeof(cufftDoubleComplex) * (N/2 + 1)  ));
    cufftDoubleComplex* d_Gfunc;
    CUDA_CHECK(cudaMalloc(&d_Gfunc, sizeof(cufftDoubleComplex) * (N/2 + 1) ) );
    get_green(d_Gfunc, N);
    CUDA_CHECK(cudaMemcpy(Gfunc, d_Gfunc, 
	    sizeof(cufftDoubleComplex) * (N/2 + 1), cudaMemcpyDeviceToHost));
    std::string fpath = "Gfunc.out";

    double dk = 1.;
    std::ofstream fcomplex(fpath);
    if (!fcomplex) {
        std::cerr << "Error: cannot open file\n" << fpath << std::endl;
    }
    fcomplex << std::scientific << std::setprecision(8);
    for (int i = 0; i < N/2+1; i++) {
        int ik = fft_freq(i, N);
        double k = abs(ik * dk);
        double re = Gfunc[i].x;
        double im = Gfunc[i].y;
        fcomplex << k << " " << re << " " << im << "\n";
    }
    fcomplex.close();
    cleanup_model();
}
