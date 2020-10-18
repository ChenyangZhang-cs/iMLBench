/**
 * 2DConvolution.c: This file is part of the PolyBench/GPU 1.0 test suite.
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
#include <string.h>
#include <omp.h>

#include <CL/cl.h>

#include "./common/polybenchUtilFuncts.h"

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 1.05

#define MAX_SOURCE_SIZE (0x100000)

/* Problem size */
#define NI 8192
#define NJ 8192

/* Thread block dimensions */
#define DIM_LOCAL_WORK_GROUP_X 32
#define DIM_LOCAL_WORK_GROUP_Y 8

#if defined(cl_khr_fp64) // Khronos extension available?
//#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64) // AMD extension available?
//#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

char str_temp[1024];

int cpu_offset = 0;
int loops = 1;
cl_platform_id platform_id;
cl_device_id device_id;
cl_uint num_devices;
cl_uint num_platforms;
cl_int errcode;
cl_context clGPUContext;
cl_kernel clKernel;
cl_command_queue clCommandQue;
cl_program clProgram;
DATA_TYPE *a_mem_obj;
DATA_TYPE *b_mem_obj;
DATA_TYPE *c_mem_obj;
FILE *fp;
char *source_str;
size_t source_size;

void Convolution2D_omp(DATA_TYPE *A, DATA_TYPE *B, int ni, int nj, size_t *cpu_global_size)
{
    // int j = get_global_id(0);
    //int i = get_global_id(1);
    DATA_TYPE c11, c12, c13, c21, c22, c23, c31, c32, c33;
    c11 = +0.2;
    c21 = +0.5;
    c31 = -0.8;
    c12 = -0.3;
    c22 = +0.6;
    c32 = -0.9;
    c13 = +0.4;
    c23 = +0.7;
    c33 = +0.10;
//#pragma omp parallel for

    for (int i = 1; i < NJ-1; i++) // 1
    {
//#pragma omp simd
        for (int j = 1; j < cpu_global_size[0]; j++) // 0
        {
            B[i * NJ + j] = c11 * A[(i - 1) * NJ + (j - 1)] + c12 * A[(i + 0) * NJ + (j - 1)] + c13 * A[(i + 1) * NJ + (j - 1)] + c21 * A[(i - 1) * NJ + (j + 0)] + c22 * A[(i + 0) * NJ + (j + 0)] + c23 * A[(i + 1) * NJ + (j + 0)] + c31 * A[(i - 1) * NJ + (j + 1)] + c32 * A[(i + 0) * NJ + (j + 1)] + c33 * A[(i + 1) * NJ + (j + 1)];
        }
    }
}

void compareResults(DATA_TYPE *B, DATA_TYPE *B_outputFromGpu)
{
    int i, j, fail;
    fail = 0;

    // Compare a and b
    for (i = 1; i < (NI - 1); i++)
    {
        for (j = 1; j < (NJ - 1); j++)
        {
            if (percentDiff(B[i * NJ + j], B_outputFromGpu[i * NJ + j]) > PERCENT_DIFF_ERROR_THRESHOLD)
            {
                printf("Fail %d, %d\n", i, j);
                fail++;
            }
        }
    }

    // Print results
    printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

void read_cl_file()
{
    // Load the kernel source code into the array source_str
    fp = fopen("2DConvolution.cl", "r");
    if (!fp)
    {
        fprintf(stdout, "Failed to load kernel.\n");
        exit(1);
    }
    source_str = (char *)malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);
}

void init(DATA_TYPE *A)
{
    int i, j;

    for (i = 0; i < NI; ++i)
    {
        for (j = 0; j < NJ; ++j)
        {
            A[i * NJ + j] = (float)rand() / RAND_MAX;
        }
    }
}

void cl_initialization()
{

    // Get platform and device information
    errcode = clGetPlatformIDs(1, &platform_id, &num_platforms);
    if (errcode == CL_SUCCESS)
        printf("number of platforms is %d\n", num_platforms);
    else
        printf("Error getting platform IDs\n");

    errcode = clGetPlatformInfo(platform_id, CL_PLATFORM_NAME, sizeof(str_temp), str_temp, NULL);
    if (errcode == CL_SUCCESS)
        printf("platform name is %s\n", str_temp);
    else
        printf("Error getting platform name\n");

    errcode = clGetPlatformInfo(platform_id, CL_PLATFORM_VERSION, sizeof(str_temp), str_temp, NULL);
    if (errcode == CL_SUCCESS)
        printf("platform version is %s\n", str_temp);
    else
        printf("Error getting platform version\n");

    errcode = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &num_devices);
    if (errcode == CL_SUCCESS)
        printf("number of devices is %d\n", num_devices);
    else
        printf("Error getting device IDs\n");

    errcode = clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(str_temp), str_temp, NULL);
    if (errcode == CL_SUCCESS)
        printf("device name is %s\n", str_temp);
    else
        printf("Error getting device name\n");

    // Create an OpenCL context
    clGPUContext = clCreateContext(NULL, 1, &device_id, NULL, NULL, &errcode);
    if (errcode != CL_SUCCESS)
        printf("Error in creating context\n");

    //Create a command-queue
    clCommandQue = clCreateCommandQueue(clGPUContext, device_id, 0, &errcode);
    if (errcode != CL_SUCCESS)
        printf("Error in creating command queue\n");
}


void cl_load_prog()
{
    // Create a program from the kernel source
    clProgram = clCreateProgramWithSource(clGPUContext, 1, (const char **)&source_str, (const size_t *)&source_size, &errcode);

    if (errcode != CL_SUCCESS)
        printf("Error in creating program\n");

    // Build the program
    errcode = clBuildProgram(clProgram, 1, &device_id, NULL, NULL, NULL);
    if (errcode != CL_SUCCESS)
        printf("Error in building program\n");

    // Create the OpenCL kernel
    clKernel = clCreateKernel(clProgram, "Convolution2D_kernel", &errcode);
    if (errcode != CL_SUCCESS)
        printf("Error in creating kernel\n");
    //clFinish(clCommandQue);
}

void cl_launch_kernel()
{
    double t_start, t_end;
    int ni = NI;
    int nj = NJ;

    size_t localWorkSize[2], globalWorkSize[2];
    localWorkSize[0] = DIM_LOCAL_WORK_GROUP_X;
    localWorkSize[1] = DIM_LOCAL_WORK_GROUP_Y;
    globalWorkSize[0] = (size_t)ceil(((float)NI) / ((float)DIM_LOCAL_WORK_GROUP_X)) * DIM_LOCAL_WORK_GROUP_X;
    globalWorkSize[1] = (size_t)ceil(((float)NJ) / ((float)DIM_LOCAL_WORK_GROUP_Y)) * DIM_LOCAL_WORK_GROUP_Y;

    size_t cpu_global_size[2];
    cpu_global_size[0] = cpu_offset * (size_t)ceil(((float)NI) / ((float)DIM_LOCAL_WORK_GROUP_X)) / 100 * DIM_LOCAL_WORK_GROUP_X;
    cpu_global_size[1] = globalWorkSize[1];
    size_t gpu_global_size[2];
    gpu_global_size[0] = globalWorkSize[0] - cpu_global_size[0];
    gpu_global_size[1] = globalWorkSize[1];
    size_t global_offset[2];
    global_offset[0] = cpu_global_size[0];
    global_offset[1] = 1;

    bool cpu_run = false, gpu_run = false;
    if (cpu_global_size[0] > 0)
    {
        cpu_run = true;
    }
    if (gpu_global_size[0] > 0)
    {
        gpu_run = true;
    }
    b_mem_obj = (DATA_TYPE *)clSVMAlloc(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * NI * NJ, 0);

    for (int j = 0; j < 3; j++){
    t_start = rtclock();

        if (gpu_run)
        {
            // Set the arguments of the kernel
            errcode = clSetKernelArgSVMPointer(clKernel, 0, (void *)a_mem_obj);
            errcode |= clSetKernelArgSVMPointer(clKernel, 1, (void *)b_mem_obj);
            errcode = clSetKernelArg(clKernel, 2, sizeof(int), (void *)&ni);
            errcode |= clSetKernelArg(clKernel, 3, sizeof(int), (void *)&nj);
            if (errcode != CL_SUCCESS)
                printf("Error in seting arguments\n");

            errcode = clEnqueueNDRangeKernel(clCommandQue, clKernel, 2, global_offset, gpu_global_size, localWorkSize, 0, NULL, NULL);
            t_start = rtclock();
//clFinish(clCommandQue);
            if (errcode != CL_SUCCESS)
                printf("Error in launching kernel\n");
        }
        if (cpu_run)
        {
            double t_start1 = rtclock();
            Convolution2D_omp(a_mem_obj, b_mem_obj, ni, nj, cpu_global_size);
            double t_end1 = rtclock();
            fprintf(stdout, "CPU time: %lfms\n", 1000.0 * (t_end1 - t_start1));
        }
        if (gpu_run)
        {
            clFinish(clCommandQue);
        }

    t_end = rtclock();
    
    fprintf(stdout, "time: %lf ms\n", 1000.0 * (t_end - t_start));
}
}

void cl_clean_up()
{
    // Clean up
    errcode = clFlush(clCommandQue);
    errcode = clFinish(clCommandQue);
    errcode = clReleaseKernel(clKernel);
    errcode = clReleaseProgram(clProgram);
    errcode = clReleaseCommandQueue(clCommandQue);
    errcode = clReleaseContext(clGPUContext);
    if (errcode != CL_SUCCESS)
        printf("Error in cleanup\n");
}

void conv2D(DATA_TYPE *A, DATA_TYPE *B)
{
    int i, j;
    DATA_TYPE c11, c12, c13, c21, c22, c23, c31, c32, c33;

    c11 = +0.2;
    c21 = +0.5;
    c31 = -0.8;
    c12 = -0.3;
    c22 = +0.6;
    c32 = -0.9;
    c13 = +0.4;
    c23 = +0.7;
    c33 = +0.10;

    for (i = 1; i < NI - 1; ++i) // 0
    {
        for (j = 1; j < NJ - 1; ++j) // 1
        {
            B[i * NJ + j] = c11 * A[(i - 1) * NJ + (j - 1)] + c12 * A[(i + 0) * NJ + (j - 1)] + c13 * A[(i + 1) * NJ + (j - 1)] + c21 * A[(i - 1) * NJ + (j + 0)] + c22 * A[(i + 0) * NJ + (j + 0)] + c23 * A[(i + 1) * NJ + (j + 0)] + c31 * A[(i - 1) * NJ + (j + 1)] + c32 * A[(i + 0) * NJ + (j + 1)] + c33 * A[(i + 1) * NJ + (j + 1)];
        }
    }
}

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        printf("usage: backprop <num of input elements>\n");
        exit(0);
    }
    cpu_offset = atoi(argv[1]);
//    loops = atoi(argv[1]);

    double t_start, t_end;
    int i;

    DATA_TYPE *A;
    DATA_TYPE *B;
    DATA_TYPE *B_outputFromGpu;

    cl_initialization();

    A = (DATA_TYPE *)clSVMAlloc(clGPUContext, CL_MEM_READ_WRITE, NI * NJ * sizeof(DATA_TYPE), 0);
    B = (DATA_TYPE *)malloc(NI * NJ * sizeof(DATA_TYPE));
    //	B_outputFromGpu = (DATA_TYPE*)malloc(NI*NJ*sizeof(DATA_TYPE));

    init(A);

    read_cl_file();
    a_mem_obj = A;
    cl_load_prog();
    cl_launch_kernel();
    /*
	errcode = clEnqueueReadBuffer(clCommandQue, b_mem_obj, CL_TRUE, 0, NI*NJ*sizeof(DATA_TYPE), B_outputFromGpu, 0, NULL, NULL);
*/
    B_outputFromGpu = b_mem_obj;

    if (errcode != CL_SUCCESS)
        printf("Error in reading GPU mem\n");

    t_start = rtclock();
    conv2D(A, B);
    t_end = rtclock();
    fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);
    compareResults(B, B_outputFromGpu);

    clSVMFree(clGPUContext, A);
    free(B);
    clSVMFree(clGPUContext, B_outputFromGpu);

    cl_clean_up();
    return 0;
}
