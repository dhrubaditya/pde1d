#include <iostream>
#include <fstream>
#include <cmath>
#include <cuComplex.h>
#include <math.h>
#include <cuda_runtime.h>
#include "misc.h"


__device__ double gaussian(double x, double x_0, double sigma, double A) 
{
    // Gaussian centered at x_0, width parameter sigma, amplitude A
    double arg = (x - x_0) / (sigma);
    return A * exp(-arg * arg);
}
//
__device__ cufftDoubleComplex cuCpow(cufftDoubleComplex z, double n) {
    double r = cuCabs(z);          // magnitude = sqrt(x^2 + y^2)
    double theta = atan2(z.y, z.x); // argument = phase
    double rn = pow(r, n);         // magnitude^n
    double ang = n * theta;        // n * phase

    return make_cuDoubleComplex(rn * cos(ang), rn * sin(ang));
}

//------------------------------------------------//
void clean_exit_host(const std::string &msg,
			      bool mark_incomplete){
  
  if (msg.empty()){
        std::cerr << "Exiting" << std::endl;
  }else{
        std::cerr << "Exiting: " << msg << std::endl;
  }
  if (mark_incomplete) {
    std::ofstream marker("INCOMPLETE");
    if (marker.is_open()) {
      marker << "Program exited early via clean_exit_host()." << std::endl;
      if (!msg.empty())
	marker << "Reason: " << msg << std::endl;
      marker.close();
      std::cerr << "Created marker file: INCOMPLETE" << std::endl;
    } else {
      std::cerr << "Warning: failed to create INCOMPLETE file." << std::endl;
    }
  }
    
  // Ensure all device resources are released
  cudaDeviceReset();
  
  // Exit cleanly
  exit(EXIT_SUCCESS);
}



