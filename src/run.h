#pragma once
#include <cufft.h>
#include <cuda_runtime.h>
#include <iostream>

// Run Parameters
struct RParams {
  bool run;
  bool FOURIER;
  int NITER;
  int NAVG;
  double dt;
  double TMAX;
};



