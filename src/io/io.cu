#include <iostream>
#include <fstream>
#include <iomanip>
#include <complex>
#include <sys/stat.h> 
#include <cufft.h>      
#include <sstream>
#include <string>
#include <unordered_map>
#include <functional>
#include <stdexcept>
#include <algorithm>
#include <cctype>
#include "io.h"
#include "misc.h"
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
static bool str_to_bool(const std::string& token)
{
    std::string t = token;
    std::transform(t.begin(), t.end(), t.begin(),
                   [](unsigned char c){ return std::tolower(c); });

    if (t == "1" || t == "true" || t == "yes" || t == "on")
        return true;
    if (t == "0" || t == "false" || t == "no" || t == "off")
        return false;

    throw std::invalid_argument("cannot convert \"" + token + "\" to bool");
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

    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: could not open input file " 
		  << filename << std::endl;
        exit(EXIT_FAILURE);
    }
    RParams p = {};  // zero-initialize

	// Helper map: key → lambda that sets the appropriate field.
    // Each lambda receives the raw value string (already trimmed).
    const std::unordered_map<std::string,
        std::function<void(const std::string&)>> setters = {
        {"run",      [&](const std::string& v){ p.run      = str_to_bool(v); }},
        {"FOURIER",  [&](const std::string& v){ p.FOURIER  = str_to_bool(v); }},
        {"NITER",    [&](const std::string& v){ p.NITER    = std::stoi(v); }},
        {"NAVG",     [&](const std::string& v){ p.NAVG     = std::stoi(v); }},
        {"dt",       [&](const std::string& v){ p.dt       = std::stod(v); }},
        {"ALGO",     [&](const std::string& v){ p.ALGO     = v;         }}
    };
 
	std::string line;
    while (std::getline(file, line)) {
        std::string tline = trim(line);
        if (tline.empty() || tline[0] == '#' || (tline.size() >= 2 && tline[0]=='/' && tline[1]=='/'))
            continue;                       // skip comments / blanks

        // Split at the first '=' character
        auto eqPos = tline.find('=');
        if (eqPos == std::string::npos) {
            std::cerr << "Warning: line without '=' ignored: " << line << '\n';
            continue;
        }

        std::string key   = trim(line.substr(0, eqPos));
        std::string value = trim(line.substr(eqPos + 1));

		auto it = setters.find(key);
        if (it != setters.end()) {
            try {
                it->second(value);   // invoke the appropriate parser
            } catch (const std::exception& e) {
                std::cerr << "Error parsing value for '" << key
                          << "': " << e.what() << '\n';
            }
        } else {
            std::cerr << "Warning: unknown key '" << key << "' ignored.\n";
        }
    }
//
    file.close();
    p.TMAX = p.dt * (double) p.NITER;
    return p;
}
//-------------------------------------------------
std::vector<int> read_vec_int(std::string& value){
  // read value = (x,y,z) into a integer vector (x,y,z)
  std::vector<int> temp_kk;
  std::stringstream ss(value);
  std::string segment;
  if (value.front() == '(' && value.back() == ')') {
    value = value.substr(1, value.size() - 2); // Remove '(' and ')'
    // 
    while (std::getline(ss, segment, ',')) {
      // Trim whitespace from the segment
      segment = trim(segment);
      if (!segment.empty()) {
	temp_kk.push_back(std::stoi(segment));
      }
    }
  }else{
    clean_exit_host("format vector = (x,y,z) needed", 1);
  }
  return temp_kk;
}
//-------------------------------------------------
std::vector<double> read_vec_double(std::string& value){
  // read value = (x,y,z) into a double vector (x,y,z)
  std::vector<double> temp_amp;
  std::stringstream ss(value);
  std::string segment;
  if (value.front() == '(' && value.back() == ')') {
    value = value.substr(1, value.size() - 2); // Remove '(' and ')'
    //              
    while (std::getline(ss, segment, ',')) {
      // Trim whitespace from the segment
      segment = trim(segment);
      if (!segment.empty()) {
	temp_amp.push_back(std::stod(segment));
      }
    }
  }else{
      clean_exit_host("format vector = (x,y,z) needed", 1);
  }
  return temp_amp;
}
//-------------------------------------------------
IParams read_icond(const std::string& filename) {
    IParams p;  // start with defaults

    std::ifstream in(filename);
    if (!in) {
        std::cerr << "Error: cannot open " << filename << " for reading\n";
	clean_exit_host("No input file", 1);
        return p;  // return defaults if file missing
    }
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;

        std::istringstream iss(line);
        std::string key, eq, value;
        
        // Read key, '=', and the rest of the line as 'value'
        if (!(iss >> key >> eq)) continue;
        
        // Get the rest of the line (the value part)
        std::getline(iss >> std::ws, value); 
        value = trim(value);

        if (key == "FOURIER") {
            p.FOURIER = (value == "true" || value == "1");
        }
        else if (key == "ITYPE") {
            p.ITYPE = value;
        }
        else if (key == "A") {
            p.A = std::stod(value);
        }
        else if (key == "xi") {
            p.xi = std::stod(value);
        }
        else if (key == "kmax") {
            p.kmax = std::stod(value);
        }
        else if (key == "kmin") {
            p.kmin = std::stod(value);
        }
        else if (key == "kpeak") {
            p.kpeak = std::stod(value);
        }
        else if (key == "kval") {
	  p.kval = read_vec_int(value);
	}
	else if (key == "amp") {
	  p.amp = read_vec_double(value);
	}
    }
    return p;
}
//------------------------------------------------//
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
    for (int i = 0; i < (N/2 + 1); i++) {
        double k = i * dk;
        out << k << " " << spectrum[i] << "\n";
    }

    out.close();
}

//--------------------------
void write_complex_array(cufftDoubleComplex* psik, 
			 double dk, int N, const std::string& fname)
{
    // --- Ensure "data" directory exists ---
    mkdir("data", 0755);
    std::string fpath = "data/" + fname;

    // --- 1. Write real-space data ---
    std::ofstream fcomplex(fpath);
    if (!fcomplex) {
        std::cerr << "Error: cannot open data/initcond_real.dat\n";
	clean_exit_host("write_complex_array: cannot open file", 1);
        return;
    }

    fcomplex << std::scientific << std::setprecision(8);
    for (int i = 0; i < N; i++) {
	int ik = fft_freq(i, N);
        double k = ik * dk;
        double re = psik[i].x;
        double im = psik[i].y;
        fcomplex << k << " " << re << " " << im << "\n";
    }
    fcomplex.close();
}
//----------------------------
void write_psi(const cufftDoubleComplex* psi, 
		    const cufftDoubleComplex* psik,
			const std::string& filename, 
		    double dx, double dk, int N)
{
    // --- Ensure "data" directory exists ---
    mkdir("data", 0755);

   	// --- 2. Write Fourier-space data ---
	std::string four_fname = "data/" + filename + "_fourier.dat";
	std::ofstream out_four(four_fname);
   	if (!out_four) {
       std::cerr << "Error: cannot open" << four_fname <<"\n";
       return;
   	}
  	out_four << std::scientific << std::setprecision(8);
    for (int i = 0; i < N; i++) {
		int ik = fft_freq(i, N);
       	double k = ik * dk;
       	double re = psik[i].x;
       	double im = psik[i].y;
      	out_four << k << " " << re << " " << im << "\n";
    }
   	out_four.close();
    // ---  Write real-space data ---
	std::string real_fname = "data/" + filename + "_real.dat";
	std::ofstream out_real(real_fname);
    if (!out_real) {
        std::cerr << "Error: cannot open" << real_fname <<"\n";
        return;
    }
    out_real << std::scientific << std::setprecision(8);
    for (int i = 0; i < N; i++) {
        double x = i * dx;
        out_real << x << " " << psi[i].x << " " <<psi[i].y << "\n";
    }
    out_real.close();
}
