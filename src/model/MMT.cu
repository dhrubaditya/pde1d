#include <iostream>
#include <fstream>
#include <iomanip>
#include <complex>
#include <sys/stat.h> 
#include <cufft.h>      
#include "misc.h"
#include "fft_utils.h"
#include "model.h"
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
// The Majda, McLaughlin, and Tabak model from J. Nonlin. Sci. Vol 6 pp 6 (1997)
struct MParams {
  double alpha;
  double beta;
  double nu;
};
bool mem_allocated;
FFTArray1D GradBetaPsik;
MParams h_MP;
// Define the device constant variable only once
__constant__ MParams d_MP;
// Read model parameters from file
// ------------------------------------------------------
// Host function to copy parameters to device constant memory
// ------------------------------------------------------
void copy_params_to_device(const MParams& h_Mparams)
{
  CUDA_CHECK(cudaMemcpyToSymbol(d_MP, &h_MP, sizeof(MParams)) );
}
//
void read_mparams(const char* filename){
  h_MP = {};  // zero-initialize
  
  std::ifstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Error: could not open parameter file " 
	      << filename << std::endl;
    exit(EXIT_FAILURE);
  }
  std::string line;
  while (std::getline(file, line)) {
    std::istringstream iss(line);
    std::string key, eq;
    double value;
    
    // Expected format: key = value
    if (!(iss >> key >> eq >> value)) continue; // skip malformed lines
    
    if (key == "alpha") h_MP.alpha = value;
    else if (key == "beta") h_MP.beta = value;
    else if (key == "nu") h_MP.nu = value;
    }    
    file.close();
}
//-------------------------------------------------
void test_model_param(){
  MParams h_MP1;
  CUDA_CHECK(cudaMemcpyFromSymbol(&h_MP1, d_MP, sizeof(MParams)) );
  printf("Check copied values:\n");
  std::cout << "alpha = " << h_MP1.alpha << "\n";
  std::cout << "beta  = " << h_MP1.beta  << "\n";
  std::cout << "nu^2    = " << (h_MP1.nu)*(h_MP1.nu)    << "\n";

}
//-------------------------------
void model_free_device_memory()
{
  if(mem_allocated){
    fft_free_1d(GradBetaPsik);
    mem_allocated = false;
  }else{
    clean_exit_host("model: cannot deallocate model dev mem.", 1);
  }

}
//
void setup_model(int N){
  // read model parameters from an input file MMT.in
  read_mparams("./input/MMT.in");
  // copy parameters to device 
  copy_params_to_device(h_MP);
  // allocate necessary device memory
  GradBetaPsik = fft_alloc_1d(N);
  mem_allocated = true;
}
void cleanup_model(){
  model_free_device_memory();
}
//----------------------------//
__device__ cufftDoubleComplex Green(double kk){
  // G = -I * k^{alpha} - nu * k^2 
  double Omega = pow(kk,d_MP.alpha);
  cufftDoubleComplex G;
  G.x = -d_MP.nu * kk * kk;
  G.y = -Omega;
  return G;
}
//
__global__ void add_lin_kernel(cufftDoubleComplex* Y,
			        cufftDoubleComplex* data, int N){
    // compute the linear part and add it to the second array
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int nfreqs = N / 2 + 1;
    if (i >= nfreqs) return;
    if (i == 0) return;
   //
   double kk = (double) i;
   cufftDoubleComplex G = Green(kk);
   cufftDoubleComplex GY = cuCmul(G, Y[i]);
   data[i].x = data[i].x + GY.x ;
   data[i].y = data[i].y + GY.y;
}
//-----------------------
void add_lin(FFTArray1D& Y, FFTArray1D& RHS){
  // compute the linear part and add it to the second array
  int nfreqs = Y.N / 2 + 1;
  int block = 256;
  int grid = (nfreqs + block - 1) / block;
  add_lin_kernel<<<grid, block>>>(Y.d_complex, RHS.d_complex, Y.N); 
}
//
void compute_nlin(FFTPlan1D& plan, FFTArray1D& Y, FFTArray1D& NLIN){
  copy_FFTArray(Y, GradBetaPsik); // psi(k)
  double beta = h_MP.beta;
  derivk(GradBetaPsik, -beta/4,  1); //|del|^{-\beta/4}psi(k)
  fft_inverse_inplace(plan, GradBetaPsik);
  normalize_fft(GradBetaPsik); //|del|^{-\beta/4}psi(r)
  cube_FFTArray(GradBetaPsik); //(|del|^{-\beta/4}psi(r) )^3 
  fft_forward_inplace(plan, GradBetaPsik); //(|del|^{-\beta/4}psi(r) )^3(k) 
  derivk(GradBetaPsik, -beta/4, 1); //|del|^{-\beta/4}
                                          //(|del|^{-\beta/4}psi(r) )^3(k) 
  cufftDoubleComplex mII = make_cuDoubleComplex(0.0, -1.0);
  complex_mult_FFTArray(GradBetaPsik, mII); 
  copy_FFTArray(GradBetaPsik, NLIN); //
}
// ------------------------------------------------------
// Compute RHS 
// ------------------------------------------------------
void compute_rhs(FFTPlan1D& plan, FFTArray1D& Y, FFTArray1D& RHS, double dt)
{
  compute_nlin(plan, Y, RHS);
  add_lin(Y, RHS);

}
// ------------------------------------------------------
// Energy kernel
// ------------------------------------------------------
__global__ void compute_energy_kernel(const double* Y, double* d_E)
{

}
