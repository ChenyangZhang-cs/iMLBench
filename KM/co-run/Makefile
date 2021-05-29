include ./make.config
# C compiler
NVCC = nvcc
CC = g++
CC_FLAGS = -g 


OPENCL_INC =/usr/local/cuda-8.0/targets/x86_64-linux/include/
OPENCL_DIR =/usr/local/cuda-8.0/lib64/
OPENCL_LIB =/usr/local/cuda-8.0/lib64/
#OPENCL_LIB =/usr/local/cuda-8.0/lib64/libOpenCL.so.1.0.0

kmeans: cluster.o getopt.o read_input.o kmeans_clustering.o rmse.o
	$(CC) $(KERNEL_DIM) $(CC_FLAGS) -fopenmp -lOpenCL kmeans.cpp cluster.o getopt.o read_input.o kmeans_clustering.o rmse.o -o kmeans -flto-report -I/usr/lib/jvm/java-11-openjdk-amd64/include/ -I/usr/lib/jvm/java-11-openjdk-amd64/include/linux/ -I/opt/rocm/opencl/include -L/opt/rocm/opencl/lib/x86_64 -lOpenCL

%.o: %.[ch]
	$(CC) $(CC_FLAGS) $< -c

cluster.o: cluster.c 
	$(CC) $(CC_FLAGS) cluster.c -c

getopt.o: getopt.c 
	$(CC) $(CC_FLAGS) getopt.c -c

kmeans.o: kmeans.c 
	$(CC) $(CC_FLAGS) read_input.c -c

rmse.o: rmse.c
	$(CC) $(CC_FLAGS) rmse.c -c

clean:
	rm -f *.o *~ kmeans *.linkinfo
