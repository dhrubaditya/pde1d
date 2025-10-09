# ==========================
# CUDA PDE Project Makefile
# ==========================

# CUDA compiler and host compiler
NVCC      := nvcc
CXX       := /usr/bin/g++-9

# Try to detect GPU architecture automatically (e.g., sm_86)
# Fallback to sm_70 if detection fails
ARCH      := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | awk -F. '{printf "sm_%d%d", $$1, $$2}' )
ifeq ($(ARCH),)
    ARCH := sm_70
endif

# Compilation and linking flags
NVCCFLAGS := -ccbin=$(CXX) -O3 -dc -arch=$(ARCH)
LDFLAGS   := -lcufft -lcurand -lcudart

# Executable name
TARGET    := pde1d.x

# Source and object files
SRC       := $(wildcard *.cu)
OBJ       := $(SRC:.cu=.o)

# Default rule
all: $(TARGET)

# Compile each CUDA source file with separate compilation (-dc)
%.o: %.cu
	@echo "Compiling $< -> $@ (arch=$(ARCH))"
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Link all object files into the final executable
$(TARGET): $(OBJ)
	@echo "Linking objects -> $(TARGET)"
	$(NVCC) -ccbin=$(CXX) $(OBJ) -o $@ $(LDFLAGS)

# Cleanup rules
clean:
	rm -f *.o *.ptx *.linkinfo

distclean: clean
	rm -f $(TARGET)

# Show configuration info
info:
	@echo "---------------- CUDA Build Info ----------------"
	@echo "NVCC       = $(NVCC)"
	@echo "CXX        = $(CXX)"
	@echo "ARCH       = $(ARCH)"
	@echo "NVCCFLAGS  = $(NVCCFLAGS)"
	@echo "LDFLAGS    = $(LDFLAGS)"
	@echo "SRC        = $(SRC)"
	@echo "OBJ        = $(OBJ)"
	@echo "TARGET     = $(TARGET)"
	@echo "------------------------------------------------"

.PHONY: all clean distclean info

