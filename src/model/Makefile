# ================================
# CUDA FFT Project Makefile
# ================================

# Compiler and flags
NVCC       := nvcc -ccbin=/usr/bin/g++-9
NVCCFLAGS  := -O3  

# Output binary name
TARGET     := mtest 
# Source and object files
SRC        := fft_utils.cu misc.cu MMT.cu test_MMT.cu 
OBJ        := $(SRC:.cu=.o)

# Header dependency
DEPS       := model.h fft_utils.h misc.h

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
	rm -f *.o *.cu~ *.h~ *.o $(TARGET)

# Phony targets
.PHONY: all clean
