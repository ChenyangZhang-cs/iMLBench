include ./make.config

# C compiler
NVCC = nvcc
CC = g++
CC_FLAGS = -g -O2 

backprop: backprop.o imagenet.o facetrain.o
	$(CC) $(CC_FLAGS) backprop_ocl.cpp backprop.o imagenet.o facetrain.o -o backprop -I/usr/lib/jvm/java-11-openjdk-amd64/include/ -I/usr/lib/jvm/java-11-openjdk-amd64/include/linux/ -I/opt/rocm/opencl/include -L/opt/rocm/opencl/lib/x86_64 -lOpenCL
%.o: %.[ch]
	$(CC) $(CC_FLAGS) $< -c
	
backprop.o: backprop.c 
	$(CC) $(CC_FLAGS) backprop.c -c
	
facetrain.o: facetrain.c
	$(CC) $(CC_FLAGS) facetrain.c -c

imagenet.o: imagenet.c
	$(CC) $(CC_FLAGS) imagenet.c -c

clean:
	rm -f *.o *~ backprop *.linkinfo
