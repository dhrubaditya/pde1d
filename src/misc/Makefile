# ================================
# CUDA FFT Project Makefile
# ================================

# Compiler and flags
NVCC       := nvcc -ccbin=/usr/bin/g++
NVCCFLAGS  := -O3  

# Output binary name
TARGET     := test
# Source and object files
#SRC        :=  misc.cu random.cu fft_utils.cu main.cu 
SRC        :=  gpu_helper.cu qgpu.cu 
OBJ        := $(SRC:.cu=.o)

# Header dependency
#DEPS       := fft_utils.h misc.h random.h
DEPS       := gpu_helper.h 

# CUDA libraries (cuFFT and runtime)
LIBS       := -lcufft -lcudart -lcurand

# Default target
all: $(TARGET)

# Link step
$(TARGET): $(OBJ)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LIBS)

# Compilation rule for .cu → .o
%.o: %.cu $(DEPS)
	$(NVCC) $(NVCCFLAGS) -dc $< -o $@

# Clean build artifacts
clean:
	rm -f $(TARGET) *.cu~ *.h~ *.o

# Phony targets
.PHONY: all clean
