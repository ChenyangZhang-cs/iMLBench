/**
 * 3DConvolution.c: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#include <omp.h>

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include "./polybenchUtilFuncts.h"

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 1.05

#define MAX_SOURCE_SIZE (0x100000)

/* Problem size */
#define NI 256
#define NJ 256
#define NK 256

/* Thread block dimensions */
#define DIM_LOCAL_WORK_GROUP_X 32
#define DIM_LOCAL_WORK_GROUP_Y 8

#if defined(cl_khr_fp64)  // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)  // AMD extension available?
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;


char str_temp[1024];

int cpu_offset = 0;
double total_time = 0;

cl_platform_id platform_id;
cl_device_id device_id;   
cl_uint num_devices;
cl_uint num_platforms;
cl_int errcode;
cl_context clGPUContext;
cl_kernel clKernel;
cl_command_queue clCommandQue;
cl_program clProgram;
cl_mem a_mem_obj;
cl_mem b_mem_obj;
FILE *fp;
char *source_str;
size_t source_size;


void Convolution3D_omp(DATA_TYPE* A, DATA_TYPE* B, int ni, int nj, int nk, int i) {
    DATA_TYPE c11, c12, c13, c21, c22, c23, c31, c32, c33;
    c11 = +2;
    c21 = +5;
    c31 = -8;
    c12 = -3;
    c22 = +6;
    c32 = -9;
    c13 = +4;
    c23 = +7;
    c33 = +10;
#pragma omp parallel for
    for (int k = 1; k < nk - 1; k++) {
        for (int j = 1; j < nj - 1; j++) {
            B[i * (nk * nj) + j * nk + k] = c11 * A[(i - 1) * (nk * nj) + (j - 1) * nk + (k - 1)] + c13 * A[(i + 1) * (nk * nj) + (j - 1) * nk + (k - 1)] + c21 * A[(i - 1) * (nk * nj) + (j - 1) * nk + (k - 1)] + c23 * A[(i + 1) * (nk * nj) + (j - 1) * nk + (k - 1)] + c31 * A[(i - 1) * (nk * nj) + (j - 1) * nk + (k - 1)] + c33 * A[(i + 1) * (nk * nj) + (j - 1) * nk + (k - 1)] + c12 * A[(i + 0) * (nk * nj) + (j - 1) * nk + (k + 0)] + c22 * A[(i + 0) * (nk * nj) + (j + 0) * nk + (k + 0)] + c32 * A[(i + 0) * (nk * nj) + (j + 1) * nk + (k + 0)] + c11 * A[(i - 1) * (nk * nj) + (j - 1) * nk + (k + 1)] + c13 * A[(i + 1) * (nk * nj) + (j - 1) * nk + (k + 1)] + c21 * A[(i - 1) * (nk * nj) + (j + 0) * nk + (k + 1)] + c23 * A[(i + 1) * (nk * nj) + (j + 0) * nk + (k + 1)] + c31 * A[(i - 1) * (nk * nj) + (j + 1) * nk + (k + 1)] + c33 * A[(i + 1) * (nk * nj) + (j + 1) * nk + (k + 1)];
        }
    }
}

void read_cl_file()
{
	// Load the kernel source code into the array source_str
	fp = fopen("3DConvolution.cl", "r");
	if (!fp) {
		fprintf(stderr, "Failed to load kernel.\n");
		exit(1);
	}
	source_str = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose( fp );
}


void init(DATA_TYPE* A)
{
	int i, j, k;

	for (i = 0; i < NI; ++i)
    	{
		for (j = 0; j < NJ; ++j)
		{
			for (k = 0; k < NK; ++k)
			{
				A[i*(NK * NJ) + j*NK + k] = i % 12 + 2 * (j % 7) + 3 * (k % 13);
			}
		}
	}
}


void cl_initialization()
{	
	// Get platform and device information
	errcode = clGetPlatformIDs(1, &platform_id, &num_platforms);
	if(errcode != CL_SUCCESS) printf("Error getting platform IDs\n");

	errcode = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &num_devices);
	if(errcode != CL_SUCCESS) printf("Error getting device IDs\n");

	// Create an OpenCL context
	clGPUContext = clCreateContext( NULL, 1, &device_id, NULL, NULL, &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating context\n");
 
	//Create a command-queue
	clCommandQue = clCreateCommandQueue(clGPUContext, device_id, 0, &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating command queue\n");
}


void cl_mem_init(DATA_TYPE* A, DATA_TYPE* B)
{
	a_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_ONLY, sizeof(DATA_TYPE) * NI * NJ * NK, NULL, &errcode);
	b_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * NI * NJ * NK, NULL, &errcode);
	
	if(errcode != CL_SUCCESS) printf("Error in creating buffers\n");

	errcode = clEnqueueWriteBuffer(clCommandQue, a_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * NI * NJ * NK, A, 0, NULL, NULL);
	errcode = clEnqueueWriteBuffer(clCommandQue, b_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * NI * NJ * NK, B, 0, NULL, NULL);
	if(errcode != CL_SUCCESS)printf("Error in writing buffers\n");
}


void cl_load_prog()
{
	// Create a program from the kernel source
	clProgram = clCreateProgramWithSource(clGPUContext, 1, (const char **)&source_str, (const size_t *)&source_size, &errcode);

	if(errcode != CL_SUCCESS) printf("Error in creating program\n");

	// Build the program
	errcode = clBuildProgram(clProgram, 1, &device_id, NULL, NULL, NULL);
	if(errcode != CL_SUCCESS) printf("Error in building program\n");
		
	// Create the OpenCL kernel
	clKernel = clCreateKernel(clProgram, "Convolution3D_kernel", &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating kernel\n");
	clFinish(clCommandQue);
}

int cl_launch_kernel(DATA_TYPE* A, DATA_TYPE* B) {
    double t_start, t_end;
    int ni = NI;
    int nj = NJ;
    int nk = NK;

    size_t localWorkSize[2], globalWorkSize[2];
    localWorkSize[0] = DIM_LOCAL_WORK_GROUP_X;
    localWorkSize[1] = DIM_LOCAL_WORK_GROUP_Y;
    globalWorkSize[0] = (size_t)ceil(((float)NK) / ((float)DIM_LOCAL_WORK_GROUP_X)) * DIM_LOCAL_WORK_GROUP_X;
    globalWorkSize[1] = (size_t)ceil(((float)NJ) / ((float)DIM_LOCAL_WORK_GROUP_Y)) * DIM_LOCAL_WORK_GROUP_Y;

    bool cpu_run = false, gpu_run = false;

    int cpu_ni = cpu_offset * NI / 100;
    int gpu_ni = NI - cpu_ni - 1;
    // printf("CPU ni: %d, GPU ni: %d\n", cpu_ni, gpu_ni);
    if (cpu_ni > 0) {
        cpu_run = true;
    }
    if (gpu_ni > 0) {
        gpu_run = true;
    }
    cl_event eventList;

    errcode = clFlush(clCommandQue);
    errcode = clFinish(clCommandQue);
    t_start = rtclock();

    if (gpu_run) {
        errcode =  clSetKernelArg(clKernel, 0, sizeof(cl_mem), (void *)&a_mem_obj);
        errcode |= clSetKernelArg(clKernel, 1, sizeof(cl_mem), (void *)&b_mem_obj);
        errcode |= clSetKernelArg(clKernel, 2, sizeof(int), &ni);
        errcode |= clSetKernelArg(clKernel, 3, sizeof(int), &nj);
        errcode |= clSetKernelArg(clKernel, 4, sizeof(int), &nk);
        if(errcode != CL_SUCCESS) printf("Error in seting arguments\n");
        if (errcode != CL_SUCCESS)
            printf("Error in seting arguments\n");
        for (int i = 1; i < gpu_ni; ++i) {
            errcode |= clSetKernelArg(clKernel, 5, sizeof(int), &i);
            errcode = clEnqueueNDRangeKernel(clCommandQue, clKernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, &eventList);
        }
        if (errcode != CL_SUCCESS)
            printf("Error in launching kernel\n");
    }
    if (cpu_run) {
        gpu_ni = gpu_ni < 1 ? 1: gpu_ni;
        for (int i = gpu_ni; i < NI-1; ++i) {
            Convolution3D_omp(A, B, ni, nj, nk, i);
        }
    }
    clFlush(clCommandQue);
    clFinish(clCommandQue);

    t_end = rtclock();
    total_time += 1000.0 * (t_end - t_start);
    // fprintf(stdout, "GPU time: %lf ms\n", 1000.0 * (t_end - t_start));
    return gpu_ni;
}



void cl_clean_up()
{
	// Clean up
	errcode = clFlush(clCommandQue);
	errcode = clFinish(clCommandQue);
	errcode = clReleaseKernel(clKernel);
	errcode = clReleaseProgram(clProgram);
	errcode = clReleaseMemObject(a_mem_obj);
	errcode = clReleaseMemObject(b_mem_obj);
	errcode = clReleaseCommandQueue(clCommandQue);
	errcode = clReleaseContext(clGPUContext);
	if(errcode != CL_SUCCESS) printf("Error in cleanup\n");
}


void compareResults(DATA_TYPE* B, DATA_TYPE* B_outputFromGpu)
{
	int i, j, k, fail;
	fail = 0;
	
	// Compare result from cpu and gpu...
	for (i = 3; i < NI - 3; ++i) // 0
	{
		for (j = 2; j < NJ - 1; ++j) // 1
		{
			for (k = 2; k < NK - 1; ++k) // 2
			{
				if (percentDiff(B[i*(NK * NJ) + j*NK + k], B_outputFromGpu[i*(NK * NJ) + j*NK + k]) > PERCENT_DIFF_ERROR_THRESHOLD)
				{
					fail++;
                    if (fail == 1)
                        printf("i, j, k: %d, %d, %d\n", i, j, k);
				}
			}	
		}
	}
	
	// Print results
	printf("Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);

}


void conv3D(DATA_TYPE* A, DATA_TYPE* B)
{
	int i, j, k;
	DATA_TYPE c11, c12, c13, c21, c22, c23, c31, c32, c33;

	c11 = +2;  c21 = +5;  c31 = -8;
	c12 = -3;  c22 = +6;  c32 = -9;
	c13 = +4;  c23 = +7;  c33 = +10;

	for (i = 1; i < NI - 1; ++i) // 0
	{
		for (j = 1; j < NJ - 1; ++j) // 1
		{
			for (k = 1; k < NK -1; ++k) // 2
			{
				//printf("i:%d\nj:%d\nk:%d\n", i, j, k);
				B[i*(NK * NJ) + j*NK + k] = c11 * A[(i - 1)*(NK * NJ) + (j - 1)*NK + (k - 1)]  +  c13 * A[(i + 1)*(NK * NJ) + (j - 1)*NK + (k - 1)]
					     +   c21 * A[(i - 1)*(NK * NJ) + (j - 1)*NK + (k - 1)]  +  c23 * A[(i + 1)*(NK * NJ) + (j - 1)*NK + (k - 1)]
					     +   c31 * A[(i - 1)*(NK * NJ) + (j - 1)*NK + (k - 1)]  +  c33 * A[(i + 1)*(NK * NJ) + (j - 1)*NK + (k - 1)]
					     +   c12 * A[(i + 0)*(NK * NJ) + (j - 1)*NK + (k + 0)]  +  c22 * A[(i + 0)*(NK * NJ) + (j + 0)*NK + (k + 0)]   
					     +   c32 * A[(i + 0)*(NK * NJ) + (j + 1)*NK + (k + 0)]  +  c11 * A[(i - 1)*(NK * NJ) + (j - 1)*NK + (k + 1)]  
					     +   c13 * A[(i + 1)*(NK * NJ) + (j - 1)*NK + (k + 1)]  +  c21 * A[(i - 1)*(NK * NJ) + (j + 0)*NK + (k + 1)]  
					     +   c23 * A[(i + 1)*(NK * NJ) + (j + 0)*NK + (k + 1)]  +  c31 * A[(i - 1)*(NK * NJ) + (j + 1)*NK + (k + 1)]  
					     +   c33 * A[(i + 1)*(NK * NJ) + (j + 1)*NK + (k + 1)];
			}
		}
	}
}


int main(int argc, char* argv[]) {
    if (argc != 2) {
        printf("usage: 3D <number of cpu offset (0~100)>\n");
        exit(0);
    }
    cpu_offset = atoi(argv[1]);
    printf("CPU offset: %d\n", cpu_offset);
	double t_start, t_end;

	DATA_TYPE* A;
	DATA_TYPE* B;
	DATA_TYPE* B_outputFromGpu;

	A = (DATA_TYPE*)malloc(NI*NJ*NK*sizeof(DATA_TYPE));
	B = (DATA_TYPE*)malloc(NI*NJ*NK*sizeof(DATA_TYPE));
	B_outputFromGpu = (DATA_TYPE*)malloc(NI*NJ*NK*sizeof(DATA_TYPE));

	int i;
	init(A);
	read_cl_file();
	cl_initialization();
	cl_mem_init(A, B);
	cl_load_prog();

	int gpu_ni = cl_launch_kernel(A, B_outputFromGpu);
    printf("Total time: %lf ms\n", total_time);
    if (cpu_offset < 100){
        errcode = clEnqueueReadBuffer(clCommandQue, b_mem_obj, CL_TRUE, 0, gpu_ni * NJ * NK * sizeof(DATA_TYPE), B_outputFromGpu, 0, NULL, NULL);
	    if(errcode != CL_SUCCESS) printf("Error in reading GPU mem\n");
    }
	

	conv3D(A, B); 
	compareResults(B, B_outputFromGpu);
	cl_clean_up();

	free(A);
	free(B);
	free(B_outputFromGpu);

	return 0;
}
