include ../make.config

LOCAL_CC = gcc -g -O3 -Wall
CC := $(CUDA_DIR)/bin/nvcc

all : nn 

clean :
	rm -rf *.o nn nno

nn : nn_cuda.cu
	$(CC) -cuda nn_cuda.cu
	$(CC) -o nn nn_cuda.cu

nno : nn_cuda_o.cu
	$(CC) -cuda nn_cuda_o.cu
	$(CC) -o nno nn_cuda_o.cu

clang: $(SRC)
	clang++ nn_cuda.cu -o nn -I../util --cuda-gpu-arch=sm_20 \
		-L/usr/local/cuda/lib64 -lcudart_static -ldl -lrt -pthread -DTIMING
