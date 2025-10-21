#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <curand_kernel.h>
#include "fft_utils.h"
#include "misc.h"
#include "initcond.h"
#include "io.h"
//-----------------
//
void set_initcond_fourier(FFTArray1D& d_psi, double dk, IParams& IC){
  unsigned long long seed = static_cast<unsigned long long>(time(nullptr));
  if (IC.ITYPE == "power") {
    std::cout << "Implementing power law spectrum with:" << std::endl;
    std::cout << "Amplitude = " << IC.A << std::endl;
    std::cout << "exponent = " << IC.xi << std::endl;
    std::cout << "between, kmin = " << IC.kmax << std::endl;
    std::cout << "and kmax = " << IC.kmin << std::endl;
    set_power_law_spectrum(d_psi, 
			   IC.A, IC.xi,
			   IC.kmin, IC.kmax,
			   seed);  
  }
  else if (IC.ITYPE == "peak") {
    std::cout << "Implementing spectrum with peak:" << std::endl;
    std::cout << "at kpeak = " << IC.kpeak << std::endl;
    std::cout << "amplitude = " << IC.A << std::endl;
    set_peak_spectrum(d_psi,
		      IC.A, dk,
		      IC.kpeak,
		      seed, 1);
  }
  else if (IC.ITYPE == "zero") {
    set_zero(d_psi);
  }
  else {
    std::cerr << "Error: initcond_fourier : unknown initial condition "
	      << IC.ITYPE << "\n";
    std::cerr << "       Terminating cleanly.\n";
    std::exit(EXIT_FAILURE);  // clean exit with failure status
  }
}
//
void set_initcond_real(FFTArray1D& d_psi, double dx, IParams& IC){
  unsigned long long seed = static_cast<unsigned long long>(time(nullptr));
  if (IC.ITYPE == "sine") {
    std::cout << "Implementing sine in real space:" << std::endl;
    std::cout << "at kpeak = " << IC.kpeak << std::endl;
    std::cout << "amplitude = " << IC.A << std::endl;
    set_sine_real(d_psi.d_real, dx,  
		  IC.A, IC.kpeak, d_psi.N);
    }
  else if (IC.ITYPE == "zero") {
    std::cout << "setting array to zero in real space:" << std::endl;
    set_zero(d_psi);
  }
  else {
    std::cerr << "Error: initcond_real : unknown initial condition "
	      << IC.ITYPE << "\n";
    std::cerr << "       Terminating cleanly.\n";
    std::exit(EXIT_FAILURE);  // clean exit with failure status
  }
}
//--------------------
void set_initcond(FFTArray1D& d_psi, double dk, double dx, IParams IC){
  std::cout << "Initial condition:" << "\t"  << IC.ITYPE  << std::endl;
  std::cout << "FOURIER:" << "\t"  << IC.FOURIER  << std::endl;
  if (IC.FOURIER){
     set_initcond_fourier(d_psi, dk, IC);
  }else{
     set_initcond_real(d_psi, dx, IC);
  }
  std::cout << "..done" << std::endl;
}
