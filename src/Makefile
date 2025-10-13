# ==========================
# CUDA PDE Project Makefile
# ==========================

# CUDA compiler and host compiler
NVCC      := nvcc
CXX       := /usr/bin/g++-9

# Compilation and linking flags
NVCCFLAGS := -ccbin=$(CXX) -O3 -dc
LDFLAGS   := -lcufft -lcurand -lcudart

# Executable name
RUN    := pde1d.x
START  := start.x

# Source and object files for START
SRC_START = start.cu initcond.cu misc.cu fft_utils.cu
OBJ_START = $(SRC_START:.cu=.o)

# Default rule
start: $(START)
run: $(RUN)

# Compile each CUDA source file with separate compilation (-dc)
%.o: %.cu
	@echo "Compiling $< -> $@ "
	$(NVCC) $(NVCCFLAGS)  $< -o $@

# Link all object files into the final executable
$(START): $(OBJ_START)
	@echo "Linking objects -> $(START)"
	$(NVCC) -ccbin=$(CXX) $(OBJ_START) -o $@ $(LDFLAGS)

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

