PROJ=linear

CC=g++

CFLAGS = -std=c99 -Wall -DUNIX -g -DDEBUG 
PROC_TYPE = $(strip $(shell uname -m | grep 64))
 
LIBS=-lOpenCL
ifeq ($(PROC_TYPE),)
	CFLAGS+=-m32
else
	CFLAGS+=-m64
endif


INC_DIRS = /opt/rocm/opencl/include 
LIB_DIRS = /opt/rocm/opencl/lib/x86_64 


SRC_FILE = linear linear_iterative linear_parallelized

all: mkdir $(PROJ)

$(PROJ): $(SRC_FILE:%=build/%.o)
	$(CC) $(CFLAGS) -o build/$@ $^ $(INC_DIRS:%=-I%) $(LIB_DIRS:%=-L%) $(LIBS)

build/linear.o: src/linear.c src/linear.h
	$(CC) $(CFLAGS) -c -o $@ $< $(INC_DIRS:%=-I%) $(LIB_DIRS:%=-L%)

build/linear_iterative.o: src/linear_iterative.c src/linear.h
	$(CC) $(CFLAGS) -c -o $@ $< $(INC_DIRS:%=-I%) $(LIB_DIRS:%=-L%)

build/linear_parallelized.o: src/linear_parallelized.c src/linear.h
	$(CC) $(CFLAGS) -c -o $@ $< $(INC_DIRS:%=-I%) $(LIB_DIRS:%=-L%)

.PHONY: clean mkdir term graph

clean:
	rm -r ./build

mkdir:
	mkdir -p build

term:
	./build/linear

graph:
	python3 graph.py home &
	python3 graph.py temperature &
