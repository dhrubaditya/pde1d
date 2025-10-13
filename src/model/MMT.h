#pragma once
#include <cufft.h>
#include <cuda_runtime.h>
#include <iostream>
void copy_params_to_device(const MParams& h_Mparams);
ModelDeviceData Model_allocate_device_memory(int N);
void Model_free_device_memory(ModelDeviceData& dev);
void setup_model();{
  // read model parameters from an input file MMT.in
  // copy parameters to device 
  // allocate necessary device memory
}
void cleanup_model(){
  // free device memory
}
// ------------------------------------------------------
// Compute RHS 
// ------------------------------------------------------
void compute_rhs(FFTPlan1D& plan, FFTArray1D& Y, FFTArray1D& RHS, double dt)
