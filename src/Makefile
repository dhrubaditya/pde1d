# ==========================
# CUDA PDE Project Makefile
# ==========================
# Disable built-in implicit rules
.SUFFIXES:
MAKEFLAGS += --no-builtin-rules
# CUDA compiler and host compiler
NVCC      := nvcc
CXX       := g++ 

# Compilation and linking flags
NVCCFLAGS := -ccbin=$(CXX) -O3 -dc
LDFLAGS   := -lcufft -lcurand -lcudart

# Executable name
RUN    := run.x
START  := start.x
E2E  := e2e.x

# Source and object files for START
SRC_START = start.cu initcond.cu misc.cu fft_utils.cu io.cu gpu_helper.cu
OBJ_START = $(SRC_START:.cu=.o)
SRC_E2E = end2end.cu initcond.cu misc.cu fft_utils.cu model.cu io.cu 
OBJ_E2E = $(SRC_E2E:.cu=.o)
SRC_RUN = run.cu initcond.cu misc.cu fft_utils.cu model.cu io.cu evolve.cu gpu_helper.cu
OBJ_RUN = $(SRC_RUN:.cu=.o)

# Default rule
start: $(START)
e2e: $(E2E)
run: $(RUN)

# Compile each CUDA source file with separate compilation (-dc)
%.o: %.cu
	@echo "Compiling $< -> $@ "
	$(NVCC) $(NVCCFLAGS)  $< -o $@

# Link all object files into the final executable
$(START): $(OBJ_START)
	@echo "Linking objects -> $(START)"
	$(NVCC) -ccbin=$(CXX) $(OBJ_START) -o $@ $(LDFLAGS)
$(E2E): $(OBJ_E2E)
	@echo "Linking objects -> $(E2E)"
	$(NVCC) -ccbin=$(CXX) $(OBJ_E2E) -o $@ $(LDFLAGS)
$(RUN): $(OBJ_RUN)
	@echo "Linking objects -> $(RUN)"
	$(NVCC) -ccbin=$(CXX) $(OBJ_RUN) -o $@ $(LDFLAGS)

# Cleanup rules
clean:
	rm -f *.o *.ptx *.linkinfo

distclean: clean
	rm -f start.x run.x e2e.x 

# Show configuration info
info:
	@echo "---------------- CUDA Build Info ----------------"
	@echo "NVCC       = $(NVCC)"
	@echo "CXX        = $(CXX)"
	@echo "ARCH       = $(ARCH)"
	@echo "NVCCFLAGS  = $(NVCCFLAGS)"
	@echo "LDFLAGS    = $(LDFLAGS)"
	@echo "SRC        = $(SRC_RUN)"
	@echo "OBJ        = $(OBJ_RUN)"
	@echo "------------------------------------------------"

.PHONY: all clean distclean info

