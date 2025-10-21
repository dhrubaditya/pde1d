#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cstdio>
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
    int N = 128;              // number of real samples
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
     cleanup_model();
}
