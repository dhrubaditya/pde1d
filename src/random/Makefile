# Makefile for CUDA random number generator project (double precision)

NVCC = nvcc -ccbin=/usr/bin/g++-9
CFLAGS = -O3 
OBJS = main.o random.o

all: main

main: $(OBJS)
	$(NVCC) $(CFLAGS) -o $@ $(OBJS) -lcurand

main.o: main.cu random.h
	$(NVCC) $(CFLAGS) -c main.cu

random.o: random.cu random.h
	$(NVCC) $(CFLAGS) -c random.cu

clean:
	rm -f *.o main random_numbers.txt

