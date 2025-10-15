#include <iostream>
#include <fstream>
#include <iomanip>
#include <complex>
#include <sys/stat.h> 
#include <cufft.h>      
#include "io.h"
//-----------
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error %s:%d: '%s'\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                               \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)
// ************************************* //
static inline std::string trim(const std::string& s) {
    size_t start = s.find_first_not_of(" \t");
    size_t end = s.find_last_not_of(" \t");
    if (start == std::string::npos) return "";
    return s.substr(start, end - start + 1);
}
//----------------------------------//
SParams read_Sparams(const char* filename) {
    SParams p = {};  // zero-initialize

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

        if (key == "NX") p.NX = static_cast<int>(value);
	else if (key == "LL") p.LL = value;
    }    
    file.close();
    p.LL = p.LL * 2 * M_PI ;// L is given in units of pi
    p.DK = 2 * M_PI / p.LL ;
    p.DX = p.LL/p.NX;
    return p;
}
//-------------------------------------------------
RParams read_Rparams(const char* filename) {
    RParams p = {};  // zero-initialize

    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: could not open input file " 
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

        if (key == "run") p.run = value;
	else if (key == "FOURIER") p.FOURIER = value;
	else if (key == "NITER") p.NITER = value;
	else if (key == "NAVG") p.NAVG = value;
	else if (key == "dt") p.dt = value;
    }    
    file.close();
    p.TMAX = dt * (double) NITER;
    return p;
}
//-------------------------------------------------
IParams read_icond(const std::string& filename) {
    IParams p;  // start with defaults

    std::ifstream in(filename);
    if (!in) {
        std::cerr << "Error: cannot open " << filename << " for reading\n";
        return p;  // return defaults if file missing
    }

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;  // skip empty and comments

        std::istringstream iss(line);
        std::string key, eq, value;
        if (!(iss >> key >> eq >> value)) continue;

        // trim possible whitespace around key/value
        /*auto trim = [](std::string s) {
            size_t start = s.find_first_not_of(" \t");
            size_t end = s.find_last_not_of(" \t");
            return (start == std::string::npos) ? std::string() : s.substr(start, end - start + 1);
        };*/
        key = trim(key);
        value = trim(value);

        if (key == "FOURIER") p.FOURIER = (value == "true" || value == "1");
        else if (key == "ITYPE") p.ITYPE = value;
        else if (key == "A") p.A = std::stod(value);
        else if (key == "xi") p.xi = std::stod(value);
        else if (key == "kmax") p.kmax = std::stod(value);
        else if (key == "kmin") p.kmin = std::stod(value);
        else if (key == "kpeak") p.kpeak = std::stod(value);
    }
    return p;
}
//


void write_spectrum(const double* spectrum, int N, double dk, int Q)
{
    // Ensure the directory exists
    mkdir("data", 0755);

    // Construct the filename: data/spec<Q>.dat
    std::ostringstream fname;
    fname << "data/spec" << Q << ".dat";

    std::ofstream out(fname.str());
    if (!out) {
        std::cerr << "Error: cannot open " << fname.str() << " for writing\n";
        return;
    }

    out << std::scientific << std::setprecision(8);
    for (int i = 0; i < N; i++) {
        double k = i * dk;
        out << k << " " << spectrum[i] << "\n";
    }

    out.close();
}

//
void write_initcond(const double* psi, const double* psik, double dx,
		    double dk, int N)
{
    // --- Ensure "data" directory exists ---
    mkdir("data", 0755);

    // --- 1. Write real-space data ---
    std::ofstream out_real("data/initcond_real.dat");
    if (!out_real) {
        std::cerr << "Error: cannot open data/initcond_real.dat\n";
        return;
    }

    out_real << std::scientific << std::setprecision(8);
    for (int i = 0; i < N; i++) {
        double x = i * dx;
        out_real << x << " " << psi[i] << "\n";
    }
    out_real.close();

    // --- 2. Write Fourier-space data ---
    std::ofstream out_four("data/initcond_fourier.dat");
    if (!out_four) {
        std::cerr << "Error: cannot open data/initcond_fourier.dat\n";
        return;
    }

    out_four << std::scientific << std::setprecision(8);
    const cufftDoubleComplex* psik_c = reinterpret_cast<const cufftDoubleComplex*>(psik);

    for (int i = 0; i < N/2 + 1; i++) {
        double k = i * dk;
        double re = psik_c[i].x;
        double im = psik_c[i].y;
        out_four << k << " " << re << " " << im << "\n";
    }
    out_four.close();
}
