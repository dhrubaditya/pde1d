#include <iostream>
#include <fstream>
#include <iomanip>
#include <complex>
#include <sys/stat.h> 
#include <cufft.h>      

struct IParams {
    bool FOURIER = true;
    std::string ITYPE = "zero";
    double A = 0.0;
    double xi = 0.0;
    int kmax = 0;
    int kmin = 0;
    int  kpeak = 1;
};


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
        auto trim = [](std::string s) {
            size_t start = s.find_first_not_of(" \t");
            size_t end = s.find_last_not_of(" \t");
            return (start == std::string::npos) ? std::string() : s.substr(start, end - start + 1);
        };
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
void set_initcond(IParams& IC){
  unsigned long long seed = static_cast<unsigned long long>(time(nullptr));
  if (IC.ITYPE == "power") {
    std::cout << "calling POWER" << std::endl;
    }
  else if (IC.ITYPE == "peak") {
    std::cout << "calling PEAK" << std::endl;

  }
  else if (IC.ITYPE == "zero") {
    std::cout << "calling ZERO" << std::endl;

  }
  else {
    std::cerr << "unknown initial condition "
	      << IC.ITYPE << "\n";
    std::cerr << "       Terminating cleanly.\n";
    std::exit(EXIT_FAILURE);  // clean exit with failure status
  }
}

int main(){
  // Read a structure from a file.
  // depending on a character in the structure
  // take a decision.
  IParams IC = read_icond("icond.in");
  std::cout << "Initial Condition:" << std::endl;
  std::cout << "FOURIER = " << IC.FOURIER << std::endl;
  std::cout << "ITYPE = " << IC.ITYPE << std::endl;
  std::cout << "A = " << IC.A << std::endl;
  std::cout << "xi = " << IC.xi << std::endl;
  std::cout << "kmax = " << IC.kmax << std::endl;
  std::cout << "kmin = " << IC.kmin << std::endl;
  std::cout << "kpeak = " << IC.kpeak << std::endl;
  if (IC.FOURIER){
  set_initcond(IC);
  }else{
      std::cout << "real space not coded yet" << std::endl;
  }
}
