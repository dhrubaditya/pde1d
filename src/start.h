#pragma once
#include <cufft.h>
#include <cuda_runtime.h>
#include <iostream>

// Start Parameters
struct SParams {
  int NX;
  double LL;
  double DX;
  double DK;
};



