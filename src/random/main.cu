#include <iostream>
#include <fstream>
#include <cstring>
#include <ctime>
#include "random.h"

int main(int argc, char **argv)
{
    if (argc < 2) {
        std::cerr << "Usage:\n";
        std::cerr << "  Uniform: ./main uni [seed N] or ./main uni [N]\n";
        std::cerr << "  Gaussian: ./main gauss <mean> <stddev> [seed N] or ./main gauss <mean> <stddev> [N]\n";
        return 1;
    }

    std::string mode = argv[1];
    double mean = 0.0, stddev = 1.0;
    unsigned long long seed = static_cast<unsigned long long>(time(nullptr)); // default: time-based seed
    size_t N = 10000;  // default number of samples

    if (mode == "uni") {
        if (argc == 3) {
            // single extra argument: treat as N
            N = static_cast<size_t>(atoll(argv[2]));
        } else if (argc >= 4) {
            // two extra arguments: seed N
            seed = strtoull(argv[2], nullptr, 10);
            N = static_cast<size_t>(atoll(argv[3]));
        }
    } else if (mode == "gauss") {
        if (argc < 4) {
            std::cerr << "Usage for Gaussian: ./main gauss <mean> <stddev> [seed N] or ./main gauss <mean> <stddev> [N]\n";
            return 1;
        }
        mean = atof(argv[2]);
        stddev = atof(argv[3]);

        if (argc == 5) {
            // single extra argument: treat as N
            N = static_cast<size_t>(atoll(argv[4]));
        } else if (argc >= 6) {
            // two extra arguments: seed N
            seed = strtoull(argv[4], nullptr, 10);
            N = static_cast<size_t>(atoll(argv[5]));
        }
    } else {
        std::cerr << "Error: unknown mode '" << mode << "'. Valid choices: 'uni', 'gauss'\n";
        return 1;
    }

    // Allocate device memory
    double *d_data;
    cudaMalloc(&d_data, N * sizeof(double));

    // Initialize RNG with selected or default seed
    rng_init(seed);

    // Generate numbers explicitly based on mode
    if (mode == "uni") {
        rng_generate_uniform(d_data, N);
    } else if (mode == "gauss") {
        rng_generate_normal(d_data, N, mean, stddev);
    }

    // Copy to host
    double *h_data = new double[N];
    cudaMemcpy(h_data, d_data, N * sizeof(double), cudaMemcpyDeviceToHost);

    // Write file with header
    std::ofstream fout("random_numbers.txt");
    if (!fout) {
        std::cerr << "Error: could not open output file.\n";
        delete[] h_data;
        cudaFree(d_data);
        rng_destroy();
        return 1;
    }

    fout << "# RNG type: " << mode << "\n";
    fout << "# seed: " << seed << "\n";
    if (mode == "gauss") {
        fout << "# mean: " << mean << "\n";
        fout << "# stddev: " << stddev << "\n";
    }
    fout << "# N: " << N << "\n";

    for (size_t i = 0; i < N; ++i)
        fout << h_data[i] << "\n";

    fout.close();
    std::cout << "Generated " << N << " random numbers written to random_numbers.txt\n";

    delete[] h_data;
    cudaFree(d_data);
    rng_destroy();

    return 0;
}

