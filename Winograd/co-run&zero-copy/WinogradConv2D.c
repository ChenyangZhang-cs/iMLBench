#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include <CL/cl.h>

#include "../polybenchUtilFuncts.h"

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 1.05

#define MAX_SOURCE_SIZE (0x100000)

/* Problem size */
#define N 2048    // 若凡：改成了二维相等
// #define NI 8192
// #define NJ 8192

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

cl_platform_id platform_id;
cl_device_id device_id;   
cl_uint num_devices;
cl_uint num_platforms;
cl_int errcode;
cl_context clGPUContext;
cl_kernel clKernel;
cl_command_queue clCommandQue;
cl_program clProgram;
DATA_TYPE* a_mem_obj;
DATA_TYPE* b_mem_obj;
DATA_TYPE* c_mem_obj;
FILE *fp;
char *source_str;
size_t source_size;

int cpu_offset;

void WinogradConv2D_2x2_omp(DATA_TYPE *input, DATA_TYPE *output, DATA_TYPE *transformed_filter, size_t *cpu_global_size);


void read_cl_file()  // 若凡：修改了文件名
{
    // Load the kernel source code into the array source_str
    fp = fopen("WinogradConv2D_2x2.cl", "r");
    if (!fp) {
        fprintf(stdout, "Failed to load kernel.\n");
        exit(1);
    }
    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose( fp );
}


void init(DATA_TYPE* A)
{
    int i, j;

    for (i = 0; i < N; ++i)
    {
        for (j = 0; j < N; ++j)
        {
            A[i*N + j] = (float)rand()/RAND_MAX;
        }
    }
}


void cl_initialization()
{
    
    // Get platform and device information
    errcode = clGetPlatformIDs(1, &platform_id, &num_platforms);
    if(errcode != CL_SUCCESS) 
        printf("Error getting platform IDs\n");


    errcode = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &num_devices);
    if(errcode != CL_SUCCESS) 
        printf("Error getting device IDs\n");

    // Create an OpenCL context
    clGPUContext = clCreateContext( NULL, 1, &device_id, NULL, NULL, &errcode);
    if(errcode != CL_SUCCESS) printf("Error in creating context\n");
 
    //Create a command-queue
    clCommandQue = clCreateCommandQueue(clGPUContext, device_id, 0, &errcode);
    if(errcode != CL_SUCCESS) printf("Error in creating command queue\n");
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
    clKernel = clCreateKernel(clProgram, "WinogradConv2D_2x2_kernel", &errcode);
    if(errcode != CL_SUCCESS) printf("Error in creating kernel\n");
    clFinish(clCommandQue);
}


void cl_launch_kernel()  // 若凡：有一点改动
{
    double t_start, t_end;
    int in_map_size = N;
    int out_map_size = N - 2;  // 这里
    int tile_n = (out_map_size + 1) / 2;

    size_t localWorkSize[2], globalWorkSize[2];
    localWorkSize[0] = DIM_LOCAL_WORK_GROUP_X;
    localWorkSize[1] = DIM_LOCAL_WORK_GROUP_Y;
    globalWorkSize[0] = (size_t)ceil(((float)tile_n) / ((float)DIM_LOCAL_WORK_GROUP_X)) * DIM_LOCAL_WORK_GROUP_X;  // 这里
    globalWorkSize[1] = (size_t)ceil(((float)tile_n) / ((float)DIM_LOCAL_WORK_GROUP_Y)) * DIM_LOCAL_WORK_GROUP_Y;  // 这里

    size_t cpu_global_size[2];
    cpu_global_size[0] = cpu_offset * (size_t)ceil(((float)tile_n) / ((float)DIM_LOCAL_WORK_GROUP_X)) / 100 * DIM_LOCAL_WORK_GROUP_X;  // 这里
    cpu_global_size[1] = globalWorkSize[1];
    size_t gpu_global_size[2];
    gpu_global_size[0] = globalWorkSize[0] - cpu_global_size[0];
    gpu_global_size[1] = globalWorkSize[1];
    size_t global_offset[2];
    global_offset[0] = cpu_global_size[0];
    // global_offset[1] = 1;
    global_offset[1] = 0;  // 这里

    bool cpu_run = false, gpu_run = false;
    if (cpu_global_size[0] > 0)
    {
        cpu_run = true;
    }
    if (gpu_global_size[0] > 0)
    {
        gpu_run = true;
    }

    t_start = rtclock();
    cl_event kernelEvent1;
    if (gpu_run)
    {
        // Set the arguments of the kernel  
        // errcode =  clSetKernelArg(clKernel, 0, sizeof(cl_mem), (void *)&a_mem_obj);
        // errcode |= clSetKernelArg(clKernel, 1, sizeof(cl_mem), (void *)&b_mem_obj);
        // errcode |= clSetKernelArg(clKernel, 2, sizeof(cl_mem), (void *)&c_mem_obj);  // 这里
        errcode =  clSetKernelArgSVMPointer(clKernel, 0, (void *)a_mem_obj);
        errcode |= clSetKernelArgSVMPointer(clKernel, 1, (void *)b_mem_obj);
        errcode |= clSetKernelArgSVMPointer(clKernel, 2, (void *)c_mem_obj);  // 这里      
        errcode |=  clSetKernelArg(clKernel, 3, sizeof(int), &in_map_size);  // 这里
        errcode |= clSetKernelArg(clKernel, 4, sizeof(int), &out_map_size);  // 这里
        if (errcode != CL_SUCCESS)
            printf("Error in seting arguments\n");
        errcode = clEnqueueNDRangeKernel(clCommandQue, clKernel, 2, global_offset, gpu_global_size, localWorkSize, 0, NULL,&kernelEvent1);
        t_start = rtclock();
        if (errcode != CL_SUCCESS)
            printf("Error in launching kernel\n");
    }
    if (cpu_run)
    {
        WinogradConv2D_2x2_omp(a_mem_obj, b_mem_obj, c_mem_obj, cpu_global_size);
    }
    if (gpu_run)
    {
        cl_int err = clWaitForEvents(1, &kernelEvent1);
        if (err != CL_SUCCESS)
            printf("ERROR in corun\n");
    }
    t_end = rtclock();

    // fprintf(stdout, "Total time: %lf ms\n", 1000.0 * (t_end - t_start));
}

void cl_clean_up()
{
    // Clean up
    errcode = clFlush(clCommandQue);
    errcode = clFinish(clCommandQue);
    errcode = clReleaseKernel(clKernel);
    errcode = clReleaseProgram(clProgram);
    // errcode = clReleaseMemObject(a_mem_obj);
    // errcode = clReleaseMemObject(b_mem_obj);
    // errcode = clReleaseMemObject(c_mem_obj);
    clSVMFree(clGPUContext, a_mem_obj);
    clSVMFree(clGPUContext, b_mem_obj);
    clSVMFree(clGPUContext, c_mem_obj);
    errcode = clReleaseCommandQueue(clCommandQue);
    errcode = clReleaseContext(clGPUContext);
    if(errcode != CL_SUCCESS) printf("Error in cleanup\n");
}


// F(2x2,3x3)

void WinogradConv2D_2x2_filter_transformation(DATA_TYPE *transformed_filter) {
    DATA_TYPE filter[3][3];

    filter[0][0] = +0.2;  filter[1][0] = +0.5;  filter[2][0] = -0.8;
    filter[0][1] = -0.3;  filter[1][1] = +0.6;  filter[2][1] = -0.9;
    filter[0][2] = +0.4;  filter[1][2] = +0.7;  filter[2][2] = +0.10;

    // filter transformation

    DATA_TYPE tmp_filter[4][3];

    // const float G[4][3] = {
    //     {1.0f, 0.0f, 0.0f},
    //     {0.5f, 0.5f, 0.5f},
    //     {0.5f, -0.5f, 0.5f},
    //     {0.0f, 0.0f, 1.0f}
    // };

    // G * g
    for (int j = 0; j < 3; j ++) {
        tmp_filter[0][j] = filter[0][j];
        tmp_filter[1][j] = 0.5f * filter[0][j] + 0.5f * filter[1][j] + 0.5f * filter[2][j];
        tmp_filter[2][j] = 0.5f * filter[0][j] - 0.5f * filter[1][j] + 0.5f * filter[2][j];
        tmp_filter[3][j] = filter[2][j];
    }
    // g * Gt
    for (int i = 0; i < 4; i ++) {
        transformed_filter[i * 4 + 0] = tmp_filter[i][0];
        transformed_filter[i * 4 + 1] = 0.5f * tmp_filter[i][0] + 0.5f * tmp_filter[i][1] + 0.5f * tmp_filter[i][2];
        transformed_filter[i * 4 + 2] = 0.5f * tmp_filter[i][0] - 0.5f * tmp_filter[i][1] + 0.5f * tmp_filter[i][2];
        transformed_filter[i * 4 + 3] = tmp_filter[i][2];
    }
}


void WinogradConv2D_2x2_omp(DATA_TYPE *input, DATA_TYPE *output, DATA_TYPE *transformed_filter, size_t *cpu_global_size) {
    // DATA_TYPE trasformed_filter[4][4];
    // WinogradConv2D_2x2_filter_transformation(trasformed_filter);

    int out_map_size = N - 2;
    int tile_n = (out_map_size + 1) / 2;

    #pragma omp parallel for
    for (int tile_i = 0; tile_i < cpu_global_size[0]; tile_i ++) {
        for (int tile_j = 0; tile_j < tile_n; tile_j ++) {
            // input transformation
            DATA_TYPE input_tile[4][4], tmp_tile[4][4], transformed_tile[4][4];
            for (int i = 0; i < 4; i ++) {
                for (int j = 0; j < 4; j ++) { 
                    int x = 2 * tile_i + i;
                    int y = 2 * tile_j + j;
                    if (x >= N || y >= N) {
                        input_tile[i][j] = 0;
                        continue;
                    }
                    input_tile[i][j] = input[x * N + y];
                }
            } 

            // const float Bt[4][4] = {
            //     {1.0f, 0.0f, -1.0f, 0.0f},
            //     {0.0f, 1.0f, 1.0f, 0.0f},
            //     {0.0f, -1.0f, 1.0f, 0.0f},
            //     {0.0f, 1.0f, 0.0f, -1.0f}
            // }

            // Bt * d
            // #pragma omp simd
            for (int j = 0; j < 4; j ++) {
                tmp_tile[0][j] = input_tile[0][j] - input_tile[2][j];
                tmp_tile[1][j] = input_tile[1][j] + input_tile[2][j];
                tmp_tile[2][j] = -input_tile[1][j] + input_tile[2][j];
                tmp_tile[3][j] = input_tile[1][j] - input_tile[3][j];
            }
            // d * B
            // #pragma omp simd
            for (int i = 0; i < 4; i ++) {
                transformed_tile[i][0] = tmp_tile[i][0] - tmp_tile[i][2];
                transformed_tile[i][1] = tmp_tile[i][1] + tmp_tile[i][2];
                transformed_tile[i][2] = -tmp_tile[i][1] + tmp_tile[i][2];
                transformed_tile[i][3] = tmp_tile[i][1] - tmp_tile[i][3];
            }

            // element-wise multiplication

            DATA_TYPE multiplied_tile[4][4];
            for (int i = 0; i < 4; i ++) {
                // #pragma omp simd
                for (int j = 0; j < 4; j ++) {
                    multiplied_tile[i][j] = transformed_tile[i][j] * transformed_filter[i * 4 + j];
                }
            }

            // output transformation

            DATA_TYPE tmp_tile_1[2][4], final_tile[2][2];

            // const float At[2][4] {
            //     {1.0f, 1.0f, 1.0f, 0.0f},
            //     {0.0f, 1.0f, -1.0f, -1.0f}
            // }

            // At * I
            // #pragma omp simd
            for (int j = 0; j < 4; j ++) {
                tmp_tile_1[0][j] = multiplied_tile[0][j] + multiplied_tile[1][j] + multiplied_tile[2][j];
                tmp_tile_1[1][j] = multiplied_tile[1][j] - multiplied_tile[2][j] - multiplied_tile[3][j];
            }
            // I * A
            // #pragma omp simd
            for (int i = 0; i < 2; i ++) {
                final_tile[i][0] = tmp_tile_1[i][0] + tmp_tile_1[i][1] + tmp_tile_1[i][2];
                final_tile[i][1] = tmp_tile_1[i][1] - tmp_tile_1[i][2] - tmp_tile_1[i][3];
            }

            for (int i = 0; i < 2; i ++) {
                for (int j = 0; j < 2; j ++) {
                    int x = 2 * tile_i + i;
                    int y = 2 * tile_j + j;
                    if (x >= out_map_size || y >= out_map_size) {
                        continue;
                    }
                    output[x * out_map_size + y] = final_tile[i][j];
                }
            }

        }  // for tile_i
    }  // for tile_j

}


void compareResults(DATA_TYPE* B, DATA_TYPE* B_outputFromGpu)  // 若凡：有改动
{
    int i, j, fail;
    fail = 0;
    
    // Compare a and b
    for (i=0; i < (N-2); i++) 
    {
        for (j=0; j < (N-2); j++) 
        {
            if (percentDiff(B[i*(N-2) + j], B_outputFromGpu[i*(N-2) + j]) > PERCENT_DIFF_ERROR_THRESHOLD) 
            {
                fail++;
            }
        }
    }
    
    // Print results
    printf("Error Threshold of %4.2f Percent: %d\n\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
    
}


void WinogradConv2D_2x2(DATA_TYPE *input, DATA_TYPE *output, DATA_TYPE *transformed_filter) {
    // DATA_TYPE trasformed_filter[4][4];
    // WinogradConv2D_2x2_filter_transformation(trasformed_filter);

    int out_map_size = N - 2;
    int tile_n = (out_map_size + 1) / 2;

    for (int tile_i = 0; tile_i < tile_n; tile_i ++) {
        for (int tile_j = 0; tile_j < tile_n; tile_j ++) {

            // input transformation

            DATA_TYPE input_tile[4][4], tmp_tile[4][4], transformed_tile[4][4];
            for (int i = 0; i < 4; i ++) {
                for (int j = 0; j < 4; j ++) { 
                    int x = 2 * tile_i + i;
                    int y = 2 * tile_j + j;
                    if (x >= N || y >= N) {
                        input_tile[i][j] = 0;
                        continue;
                    }
                    input_tile[i][j] = input[x * N + y];
                }
            } 

            // const float Bt[4][4] = {
            //     {1.0f, 0.0f, -1.0f, 0.0f},
            //     {0.0f, 1.0f, 1.0f, 0.0f},
            //     {0.0f, -1.0f, 1.0f, 0.0f},
            //     {0.0f, 1.0f, 0.0f, -1.0f}
            // }

            // Bt * d
            for (int j = 0; j < 4; j ++) {
                tmp_tile[0][j] = input_tile[0][j] - input_tile[2][j];
                tmp_tile[1][j] = input_tile[1][j] + input_tile[2][j];
                tmp_tile[2][j] = -input_tile[1][j] + input_tile[2][j];
                tmp_tile[3][j] = input_tile[1][j] - input_tile[3][j];
            }
            // d * B
            for (int i = 0; i < 4; i ++) {
                transformed_tile[i][0] = tmp_tile[i][0] - tmp_tile[i][2];
                transformed_tile[i][1] = tmp_tile[i][1] + tmp_tile[i][2];
                transformed_tile[i][2] = -tmp_tile[i][1] + tmp_tile[i][2];
                transformed_tile[i][3] = tmp_tile[i][1] - tmp_tile[i][3];
            }

            // element-wise multiplication

            DATA_TYPE multiplied_tile[4][4];
            for (int i = 0; i < 4; i ++) {
                for (int j = 0; j < 4; j ++) {
                    multiplied_tile[i][j] = transformed_tile[i][j] * transformed_filter[i * 4 + j];
                }
            }

            // output transformation

            DATA_TYPE tmp_tile_1[2][4], final_tile[2][2];

            // const float At[2][4] {
            //     {1.0f, 1.0f, 1.0f, 0.0f},
            //     {0.0f, 1.0f, -1.0f, -1.0f}
            // }

            // At * I
            for (int j = 0; j < 4; j ++) {
                tmp_tile_1[0][j] = multiplied_tile[0][j] + multiplied_tile[1][j] + multiplied_tile[2][j];
                tmp_tile_1[1][j] = multiplied_tile[1][j] - multiplied_tile[2][j] - multiplied_tile[3][j];
            }
            // I * A
            for (int i = 0; i < 2; i ++) {
                final_tile[i][0] = tmp_tile_1[i][0] + tmp_tile_1[i][1] + tmp_tile_1[i][2];
                final_tile[i][1] = tmp_tile_1[i][1] - tmp_tile_1[i][2] - tmp_tile_1[i][3];
            }

            for (int i = 0; i < 2; i ++) {
                for (int j = 0; j < 2; j ++) {
                    int x = 2 * tile_i + i;
                    int y = 2 * tile_j + j;
                    if (x >= out_map_size || y >= out_map_size) {
                        continue;
                    }
                    output[x * out_map_size + y] = final_tile[i][j];
                }
            }

        }  // for tile_i
    }  // for tile_j

}


int main(int argc, char *argv[]) 
{
    if (argc != 2) 
    {
        printf("usage: ./WinogradConv2D <cpu offset>\n");
        exit(0);
    }
    cpu_offset = atoi(argv[1]);
    printf("CPU offset: %d\n", cpu_offset);

    double t_start, t_end;
    int i;

    DATA_TYPE *B = (DATA_TYPE*)malloc((N-2)*(N-2)*sizeof(DATA_TYPE));

    read_cl_file();
    cl_initialization();
    cl_load_prog();
    a_mem_obj = (DATA_TYPE *)clSVMAlloc(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * N * N, 0);
    b_mem_obj = (DATA_TYPE *)clSVMAlloc(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * (N-2) * (N-2), 0);
    c_mem_obj = (DATA_TYPE *)clSVMAlloc(clGPUContext, CL_MEM_READ_ONLY, sizeof(DATA_TYPE) * 4 * 4, 0);
    if (a_mem_obj == NULL || b_mem_obj == NULL || c_mem_obj == NULL)
        printf("clSVMAlloc failed\n");
    WinogradConv2D_2x2_filter_transformation(c_mem_obj);  // 这里
    
    init(a_mem_obj);
    t_start = rtclock();
    for (int i = 0; i < 3; i++)
        cl_launch_kernel();
    t_end = rtclock();
    fprintf(stdout, "Total time: %lf ms\n", 1000.0 * (t_end - t_start));

    WinogradConv2D_2x2(a_mem_obj, B, c_mem_obj);  
    compareResults(B, b_mem_obj);

    free(B);

    cl_clean_up();
        return 0;
}