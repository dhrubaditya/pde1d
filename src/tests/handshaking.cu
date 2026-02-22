#include <stdio.h>
#include <stdlib.h>
#include <cufft.h>
#include <cuComplex.h>
#include <time.h>
#include <math.h>

struct FFTArray1D {
    int N;
    cufftDoubleComplex* d_complex = nullptr;
    bool IsFourier = true;
};

// Function to associate complex array to FFT structure
void AssociateComplex2FFT(FFTArray1D& AFFT, cufftDoubleComplex* d_A, int N) {
    AFFT.d_complex = d_A;
    AFFT.N = N;
    AFFT.IsFourier = false;
}

// Device kernel to multiply by 2
__global__ void multiplyBy2(cufftDoubleComplex* d_data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        d_data[idx].x *= 2.0;
        d_data[idx].y *= 2.0;
    }
}

// Function to copy device data back to host
void copyDeviceToHost(cufftDoubleComplex* d_A, cufftDoubleComplex* h_A, int N) {
    cudaMemcpy(h_A, d_A, N * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
}

// Function to compare two complex arrays
void compareArrays(cufftDoubleComplex* original, cufftDoubleComplex* result, int N) {
    printf("Comparing arrays...\n");
    bool allMatch = true;
    double tolerance = 1e-10;
    
    for (int i = 0; i < N; i++) {
        double expected_real = original[i].x * 2.0;
        double expected_imag = original[i].y * 2.0;
        
        double diff_real = fabs(result[i].x - expected_real);
        double diff_imag = fabs(result[i].y - expected_imag);
        
        if (diff_real > tolerance || diff_imag > tolerance) {
            printf("Mismatch at index %d: Expected (%.6f, %.6f), Got (%.6f, %.6f)\n",
                   i, expected_real, expected_imag, result[i].x, result[i].y);
            allMatch = false;
        }
    }
    
    if (allMatch) {
        printf("SUCCESS: All elements match! Array was correctly multiplied by 2.\n");
    } else {
        printf("FAILURE: Some elements do not match.\n");
    }
}

int main() {
    int N = 8;
    printf("Array size N = %d\n", N);
    
    // Allocate host array and populate with random numbers
    cufftDoubleComplex* h_A = (cufftDoubleComplex*)malloc(N * sizeof(cufftDoubleComplex));
    srand(time(NULL));
    
    printf("Populating host array with random numbers...\n");
    for (int i = 0; i < N; i++) {
        h_A[i].x = (double)rand() / RAND_MAX;
        h_A[i].y = (double)rand() / RAND_MAX;
    }
    
    // Allocate device array
    cufftDoubleComplex* d_A = nullptr;
    cudaError_t err = cudaMalloc(&d_A, N * sizeof(cufftDoubleComplex));
    if (err != cudaSuccess) {
        printf("Error allocating device memory: %s\n", cudaGetErrorString(err));
        free(h_A);
        return 1;
    }
    printf("Device memory allocated successfully.\n");
    
    // Copy host array to device
    cudaMemcpy(d_A, h_A, N * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice);
    printf("Data copied to device.\n");
    
    // Create FFTArray structure and associate with device pointer
    FFTArray1D AFFT;
    AssociateComplex2FFT(AFFT, d_A, N);
    printf("FFTArray structure created and associated with device pointer.\n");
    
    // Call kernel to multiply by 2
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    multiplyBy2<<<gridSize, blockSize>>>(AFFT.d_complex, N);
    
    // Wait for kernel to complete
    cudaDeviceSynchronize();
    printf("Kernel completed: multiplied all elements by 2.\n");
    
    // Copy result back to host
    cufftDoubleComplex* h_result = (cufftDoubleComplex*)malloc(N * sizeof(cufftDoubleComplex));
    copyDeviceToHost(d_A, h_result, N);
    printf("Result copied back to host.\n");
    
    // Compare with 2x original
    compareArrays(h_A, h_result, N);
    
    // Cleanup
    cudaFree(d_A);
    free(h_A);
    free(h_result);
    
    printf("Done.\n");
    return 0;
}
