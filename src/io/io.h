#pragma once
#include <cufft.h>
#include <cuda_runtime.h>
#include "start.h"
#include "run.h"
#include "initcond.h"

SParams read_Sparams(const char* filename);
RParams read_Rparams(const char* filename);
IParams read_icond(const std::string& filename);
void write_initcond(const double* psi,
		    const double* psik,
		    double dx,
		    double dk,
		    int N);
//void read_initcond(double* psi,
//   double* psik, int N, RParams RP);
void write_spectrum(const double* Ek, int N, double dk, int Q);
