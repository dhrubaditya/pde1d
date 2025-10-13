# ================================
# CUDA FFT Project Makefile
# ================================

# Compiler and flags
NVCC       := nvcc -ccbin=/usr/bin/g++-9
NVCCFLAGS  := -O3  

# Output binary name
TARGET      := start
# Source and object files
SRC_START        := start.cu misc.cu fft_utils.cu initcond.cu io.cu 
OBJ_START        := $(SRC_START:.cu=.o)

# Header dependency
DEPS       := start.h fft_utils.h misc.h initcond.h io.h 

# CUDA libraries (cuFFT and runtime)
LIBS       := -lcufft -lcudart -lcurand

# Default target
all: $(TARGET)

# Link step
$(TARGET): $(OBJ_START)
	$(NVCC) $(NVCCFLAGS) $(OBJ_START) -o start.x  $(LIBS)

# Compilation rule for .cu → .o
%.o: %.cu $(DEPS)
	$(NVCC) $(NVCCFLAGS) -dc $< -o $@

# Clean build artifacts
clean:
	rm -f $(OBJ_START) start.x *.cu~ *.h~ 

# Phony targets
.PHONY: all clean
