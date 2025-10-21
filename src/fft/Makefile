# ================================
# CUDA FFT Project Makefile
# ================================

# Compiler and flags
NVCC       := nvcc -ccbin=/usr/bin/g++-9
NVCCFLAGS  := -O3  

# Output binary name
#TARGET     := fft_test
#TARGET     := deriv 
#TARGET      := spec
#TARGET      := sine
TARGET      := cube 
# Source and object files
#SRC        := main.cu fft_utils.cu 
SRC        := deriv_test.cu misc.cu fft_utils.cu 
#SRC        := test_spec.cu misc.cu fft_utils.cu 
#SRC        := test_sine.cu misc.cu fft_utils.cu 
SRC        := test_cube.cu misc.cu fft_utils.cu 
OBJ        := $(SRC:.cu=.o)

# Header dependency
DEPS       := fft_utils.h misc.h

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
