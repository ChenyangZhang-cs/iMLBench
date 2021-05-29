#C compiler
CC = g++

SRC = 2DConvolution.c

CC_FLAGS = -g -O2

EXE = 2D

release:
	$(CC) $(CC_FLAGS) -lOpenCL  -fopenmp $(SRC) -o $(EXE) -I/usr/lib/jvm/java-11-openjdk-amd64/include/ -I/usr/lib/jvm/java-11-openjdk-amd64/include/linux/ -I/opt/rocm/opencl/include -L/opt/rocm/opencl/lib/x86_64 -lOpenCL

clean: 
	rm -f $(EXE) 
