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
  double Omega0 ;
  double Epsilon;
};
bool mem_allocated;
FFTPlan1D plan;
FFTArray1D NLIN;
FFTArray1D GradAlphaPsik;
double* d_Hr;
MParams h_MP;
GpuReducer red;
// Define the device constant variable only once
__constant__ MParams d_MP;
// ------------------------------------------------------
// Host function to copy parameters to device constant memory
// ------------------------------------------------------
void copy_params_to_device(const MParams& h_MP)
{
  CUDA_CHECK(cudaMemcpyToSymbol(d_MP, &h_MP, sizeof(MParams)) );
}
//
void read_mparams(const char* filename){
  h_MP = {};  // zero-initialize
  h_MP.Omega0 = 1.;
  h_MP.Epsilon = 1.;
  std::ifstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Error: could not open parameter file " 
	      << filename << std::endl;
    clean_exit_host("model input file not found",1);
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
    else if (key == "Omega0") h_MP.Omega0 = value;
    else if (key == "Epsilon") h_MP.Epsilon = value;
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
  std::cout << "nu    = " << (h_MP1.nu)  << "\n";
  std::cout << "Omega0    = " << h_MP1.Omega0  << "\n";
  std::cout << "Epsilon    = " << h_MP1.Epsilon  << "\n";

}
//-------------------------------
void setup_model(int N){
  // read model parameters from an input file MMT.in
  read_mparams("./input/MMT.in");
  // copy parameters to device 
  copy_params_to_device(h_MP);
  // allocate necessary device memory
  NLIN = fft_alloc_1d(N);
  GradAlphaPsik = fft_alloc_1d(N);
  plan = fft_plan_create_1d(N);
  size_t bytes = sizeof(double) * N;
  CUDA_CHECK(cudaMalloc(&d_Hr, bytes));
  init_reducer(red, N);
  mem_allocated = true;
}
//
void model_free_device_memory()
{
  if(mem_allocated){
    fft_free_1d(NLIN);
    fft_free_1d(GradAlphaPsik);
    mem_allocated = false;
  }else{
    clean_exit_host("model: cannot deallocate model dev mem.", 1);
  }

}
//
void cleanup_model(){
  model_free_device_memory();
  fft_plan_destroy_1d(plan);
  free_reducer(red);
}
//----------------------------//
__device__ cufftDoubleComplex Green(double kk){
  // G = -I * k^{alpha} - nu * k^2 
  double Omega = d_MP.Omega0 * pow(kk,d_MP.alpha);
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

// ------------------------------------------------------
// Hamiltonian
// ------------------------------------------------------
__global__ void compute_Hr_kernel(double* DelAlphaby2psi, double* psi4,
				  double* d_Hamil, int N){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i > N) return;
  double Omega0 = d_MP.Omega0;
  double Epsilon = d_MP.Epsilon;
  double term1 = DelAlphaby2psi[i];
  double term2 = psi4[i] ;
  d_Hamil[i] = Omega0 * term1  * term1 +
    (1./2.) * Epsilon * term2 * term2 * term2 * term2;

}
//
void compute_Hr(FFTArray1D& GradAlphaby2psi, FFTArray1D& psi4){
  if(!GradAlphaby2psi.IsFourier && !psi4.IsFourier){
    int N = GradAlphaby2psi.N;
    int block = 256;
    int grid = (N + block - 1) / block;
    compute_Hr_kernel<<<grid, block>>>(GradAlphaby2psi.d_real,
				     psi4.d_real, d_Hr, N);
  }else{
    clean_exit_host("compute_Hr: should be in real space", 1);
  } 
}
//
double Hamiltonian(FFTPlan1D& plan, FFTArray1D& psik){
  int N = psik.N;
  double alpha = h_MP.alpha;
  double beta = h_MP.beta;
  copy_FFTArray(psik, NLIN); // psi(k)
  derivk(NLIN, -beta/4,  true); //|del|^{-\beta/4}psi(k)
  fft_inverse_inplace(plan, NLIN);
  normalize_fft(NLIN); //|del|^{-\beta/4}psi(r)
  
  copy_FFTArray(psik, GradAlphaPsik); // psi(k)
  derivk(GradAlphaPsik, alpha/2,  true); //|del|^{\alpha/2}psi(k)
  fft_inverse_inplace(plan, GradAlphaPsik);
  normalize_fft(GradAlphaPsik); //|del|^{\alpha/2}psi(r)
  compute_Hr(GradAlphaPsik, NLIN);
  double HH = gpu_sum(d_Hr, N, red);
  return HH;
}
//  Transform between vv and psi 
__global__ void psik_to_vv__kernel(const cufftDoubleComplex* d_psik,
			   cufftDoubleComplex* d_vv,
			   double time, int N){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int nfreqs = N / 2 + 1;
  if (i >= nfreqs) return;
  if (i == 0) return;
  //
  if (time == 0.){
    d_vv[i] = d_psik[i];
  }else{
  double kk = (double) i;
  cufftDoubleComplex G = Green(kk);
  cufftDoubleComplex emGt = exp_cuComplex(G, -time);
  d_vv[i] = cuCmul(emGt, d_psik[i]);
  }
}
//
__global__ void vv_to_psik__kernel(const cufftDoubleComplex* d_vv,
			   cufftDoubleComplex* d_psik,
			   double time, int N){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int nfreqs = N / 2 + 1;
  if (i >= nfreqs) return;
  if (i == 0) return;
  //
  if (time == 0.){
     d_psik[i] = d_vv[i];
  }else{
  double kk = (double) i;
  cufftDoubleComplex G = Green(kk);
  cufftDoubleComplex eGt = exp_cuComplex(G, time);
  d_psik[i] = cuCmul(eGt, d_vv[i]);
  }
}
void exp_inv_transform(double* vv, double* psik, double time, int N ){
  FFTArray1D Fpsik;
  FFTArray1D Fvv;
  double2FFTArray(Fpsik, psik, N);
  double2FFTArray(Fvv, vv, N);
  int nfreqs = N / 2 + 1;
  int block = 256;
  int grid = (nfreqs + block - 1) / block;
  vv_to_psik__kernel<<<grid, block>>>(Fvv.d_complex,
				      Fpsik.d_complex,
				      time, N);
}
//
void exp_transform(double* psik, double* vv, double time, int N ){
  FFTArray1D Fpsik;
  FFTArray1D Fvv;
  double2FFTArray(Fpsik, psik, N);
  double2FFTArray(Fvv, vv, N);
  int nfreqs = N / 2 + 1;
  int block = 256;
  int grid = (nfreqs + block - 1) / block;
  psik_to_vv__kernel<<<grid, block>>>(Fpsik.d_complex,
				      Fvv.d_complex,
				      time, N);
}
//---------------------
__global__ void mult_prefactor_rhsv_kernel(cufftDoubleComplex* d_psi4,
			       cufftDoubleComplex* vrhs,
			       double time, double dt,
			       int N){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int nfreqs = N / 2 + 1;
  if (i >= nfreqs) return;
  if (i == 0) return;
  //
  cufftDoubleComplex rhs;
  double kk = (double) i;  
  if (time == 0){
    rhs.x = d_psi4[i].x ;
    rhs.y = d_psi4[i].y ;
  }else{
    cufftDoubleComplex G = Green(kk);
    cufftDoubleComplex emGt = exp_cuComplex(G, -time);
    rhs = cuCmul(emGt,d_psi4[i]);
  }
  vrhs[i].x =    dt * rhs.y / pow(kk, d_MP.beta)  ;
  vrhs[i].y = -  dt * rhs.x / pow(kk, d_MP.beta);    
}
//---------------------
void compute_nlin(const FFTArray1D& psik){
  copy_FFTArray(psik, NLIN); // NLIN = psi(k)
  double beta = h_MP.beta;
  double Epsilon = h_MP.Epsilon;
  derivk(NLIN, -beta/4,  true); //k^{-\beta/4}psi
  fft_inverse_inplace(plan, NLIN);
  normalize_fft(NLIN); //F^{-1}(k^{-\beta/4}psi)
  cube_FFTArray(NLIN); //( F^{-1}(k^{-\beta/4}psi) )^3 
  fft_forward_inplace(plan, NLIN); //
                          //F( (F^{-1}(|del|^{-\beta/4}psi) )^3 )
  derivk(NLIN, -beta/4, true); //
                          //(k^{-beta/4}F( (F^{-1}(|del|^{-\beta/4}psi) )^3 )
}
//---------------------
void compute_rhsv(const FFTArray1D& psik, FFTArray1D& rhs,
		  double time, double dt){
  compute_nlin(psik); // the nlin term is stored in NLIN 
  int N = psik.N;
  int nfreqs = N / 2 + 1;
  int block = 256;
  int grid = (nfreqs + block - 1) / block;
  mult_prefactor_rhsv_kernel<<<grid, block>>>(NLIN.d_complex,
				       rhs.d_complex, time, dt, N); 
}
//-------------------
cufftDoubleComplex compute_NN_nlin(const FFTArray1D& psik){
  int N = psik.N;
  int nfreqs = N / 2 + 1;
  int block = 256;
  int grid = (nfreqs + block - 1) / block;
  mult_Astar_B<<<grid, block>>>(psik.d_complex,
				NLIN.d_complex, N);
  GpuComplexReducer ws;
  init_Complex_reducer(ws, N);
  cufftDoubleComplex sum = gpu_Complex_sum(NLIN.d_complex,
					       N, ws);
  free_Complex_reducer(ws);
  return sum;
}
//-----------------
cufftDoubleComplex test_NN_conservation(FFTArray1D& psik){
  if (NLIN.d_complex == nullptr)
    { printf("something is wrong \n");
      clean_exit_host("FFT1DArray NLIN not allocated", 1);
    }
  compute_nlin(psik);
  cufftDoubleComplex Z = compute_NN_nlin(psik);
  return Z;
}
void copy_NLIN2host(cufftDoubleComplex* h_nlin, double* h_nlink, 
		    const FFTArray1D& d_psi)
{
  if (NLIN.d_complex == nullptr)
    { printf("copy_NLIN2host: something is wrong \n");
      clean_exit_host("FFT1DArray NLIN not allocated", 1);
    }
  compute_nlin(d_psi);
  copy_FFTArray_host_complex(h_nlin, NLIN);
  int N = d_psi.N;
  double* d_nlink;
  CUDA_CHECK(cudaMalloc(&d_nlink, sizeof(double) * (N/2 + 1)) );
  compute_spectrum(NLIN, d_nlink);
  normalize_spectrum(d_nlink, N);
  CUDA_CHECK(cudaMemcpy(h_nlink, d_nlink, sizeof(double) * (N/2 + 1),
                          cudaMemcpyDeviceToHost));
  cudaFree(d_nlink);
}
// ------------------------------------------------------
// Compute RHS 
// ------------------------------------------------------
void compute_rhs(double* d_vv, double* d_psik, double* RHS,
		 double tt, double dt, int N,
		 int stage)
{
  FFTArray1D Fvv;
  FFTArray1D Fpsik;
  FFTArray1D Frhs;
  double2FFTArray(Fvv, d_vv, N);
  double2FFTArray(Fpsik, d_psik, N);
  double2FFTArray(Frhs, RHS, N);
  compute_rhsv(Fpsik, Frhs, tt, dt); 
}
