#pragma once
#include <cufft.h>
#include <cuda_runtime.h>
#include <iostream>

// Start Parameters
struct RParams {
  bool run;
  bool FOURIER;
  int NITER;
  int NAVG;
};



