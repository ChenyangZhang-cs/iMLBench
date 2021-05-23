#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "linear.h"

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS


//local_size 109
//local_id   0-108
//group_id   0-4
//global_id  0-544  (5*109)

/* OpenCL structures */
//#define group_id_size 5
//#define local_size 109

#define group_id_size 281
#define local_size 58
static cl_device_id device;
static cl_context context;
static cl_program program;
static cl_command_queue queue;
static cl_int err;

int cpu_offset = 0;

// err |= clSetKernelArg(kernel, 3, params->wg_size * sizeof(rsquared_t), NULL);

void rsquaredOMP(data_t* dataset,
                 cl_float mean,
                 cl_float equation[2],  // [a0,a1]
                 rsquared_t* result,
                 int cpusize) {
    rsquared_t* dist = (rsquared_t*)malloc(sizeof(rsquared_t) * local_size);

    for (int group_id = 0; group_id < (cpusize / local_size); group_id++) {
        for (int loc_id = local_size - 1; loc_id >= 0; loc_id--) {
            int glob_id = loc_id + group_id * 58;
            dist[loc_id].actual = pow((dataset[glob_id].x - mean), 2);
            float y_estimated = dataset[glob_id].x * equation[0] + equation[1];
            dist[loc_id].estimated = pow((y_estimated - mean), 2);
            for (int i = (local_size / 2), old_i = local_size; i > 0; old_i = i, i /= 2) {
                if (loc_id < i) {
                    dist[loc_id].actual += dist[loc_id + i].actual;
                    dist[loc_id].estimated += dist[loc_id + i].estimated;
                    if (loc_id == (i - 1) && old_i % 2 != 0) {
                        dist[loc_id].actual += dist[old_i - 1].actual;
                        dist[loc_id].estimated += dist[old_i - 1].estimated;
                    }
                }
            }
            if (loc_id == 0) {
                result[group_id] = dist[0];
            }
        }
    }
}

void linear_regressionOMP(
    data_t* dataset,
    sum_t* result,
    int cpusize) {
    sum_t* interns = (sum_t*)malloc(sizeof(sum_t) * local_size);
    //int cpusize = local_size * group_id_size * cpu_offset / 100;

    for (int group_id = 0; group_id < (cpusize / local_size); group_id++) {
        for (int loc_id = local_size - 1; loc_id >= 0; loc_id--) {
            int glob_id = loc_id + group_id * 58;
            interns[loc_id].sumx = dataset[glob_id].x;
            interns[loc_id].sumy = dataset[glob_id].y;
            interns[loc_id].sumxy = (dataset[glob_id].x * dataset[glob_id].y);
            interns[loc_id].sumxsq = (dataset[glob_id].x * dataset[glob_id].x);
            for (
                int i = (local_size / 2), old_i = local_size;
                i > 0;
                old_i = i, i /= 2) {
                if (loc_id < i) {
                    interns[loc_id].sumx += interns[loc_id + i].sumx;
                    interns[loc_id].sumy += interns[loc_id + i].sumy;
                    interns[loc_id].sumxy += interns[loc_id + i].sumxy;
                    interns[loc_id].sumxsq += interns[loc_id + i].sumxsq;
                    if (loc_id == (i - 1) && old_i % 2 != 0) {
                        interns[loc_id].sumx += interns[old_i - 1].sumx;
                        interns[loc_id].sumy += interns[old_i - 1].sumy;
                        interns[loc_id].sumxy += interns[old_i - 1].sumxy;
                        interns[loc_id].sumxsq += interns[old_i - 1].sumxsq;
                    }
                }
            }
            if (loc_id == 0) {
                result[group_id] = interns[0];
            }
        }
    }
}

static const char* error_msg(cl_int error) {
    switch (error) {
        // run-time and JIT compiler errors
        case 0:
            return "CL_SUCCESS";
        case -1:
            return "CL_DEVICE_NOT_FOUND";
        case -2:
            return "CL_DEVICE_NOT_AVAILABLE";
        case -3:
            return "CL_COMPILER_NOT_AVAILABLE";
        case -4:
            return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case -5:
            return "CL_OUT_OF_RESOURCES";
        case -6:
            return "CL_OUT_OF_HOST_MEMORY";
        case -7:
            return "CL_PROFILING_INFO_NOT_AVAILABLE";
        case -8:
            return "CL_MEM_COPY_OVERLAP";
        case -9:
            return "CL_IMAGE_FORMAT_MISMATCH";
        case -10:
            return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case -11:
            return "CL_BUILD_PROGRAM_FAILURE";
        case -12:
            return "CL_MAP_FAILURE";
        case -13:
            return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
        case -14:
            return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
        case -15:
            return "CL_COMPILE_PROGRAM_FAILURE";
        case -16:
            return "CL_LINKER_NOT_AVAILABLE";
        case -17:
            return "CL_LINK_PROGRAM_FAILURE";
        case -18:
            return "CL_DEVICE_PARTITION_FAILED";
        case -19:
            return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

        // compile-time errors
        case -30:
            return "CL_INVALID_VALUE";
        case -31:
            return "CL_INVALID_DEVICE_TYPE";
        case -32:
            return "CL_INVALID_PLATFORM";
        case -33:
            return "CL_INVALID_DEVICE";
        case -34:
            return "CL_INVALID_CONTEXT";
        case -35:
            return "CL_INVALID_QUEUE_PROPERTIES";
        case -36:
            return "CL_INVALID_COMMAND_QUEUE";
        case -37:
            return "CL_INVALID_HOST_PTR";
        case -38:
            return "CL_INVALID_MEM_OBJECT";
        case -39:
            return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        case -40:
            return "CL_INVALID_IMAGE_SIZE";
        case -41:
            return "CL_INVALID_SAMPLER";
        case -42:
            return "CL_INVALID_BINARY";
        case -43:
            return "CL_INVALID_BUILD_OPTIONS";
        case -44:
            return "CL_INVALID_PROGRAM";
        case -45:
            return "CL_INVALID_PROGRAM_EXECUTABLE";
        case -46:
            return "CL_INVALID_KERNEL_NAME";
        case -47:
            return "CL_INVALID_KERNEL_DEFINITION";
        case -48:
            return "CL_INVALID_KERNEL";
        case -49:
            return "CL_INVALID_ARG_INDEX";
        case -50:
            return "CL_INVALID_ARG_VALUE";
        case -51:
            return "CL_INVALID_ARG_SIZE";
        case -52:
            return "CL_INVALID_KERNEL_ARGS";
        case -53:
            return "CL_INVALID_WORK_DIMENSION";
        case -54:
            return "CL_INVALID_WORK_GROUP_SIZE";
        case -55:
            return "CL_INVALID_WORK_ITEM_SIZE";
        case -56:
            return "CL_INVALID_GLOBAL_OFFSET";
        case -57:
            return "CL_INVALID_EVENT_WAIT_LIST";
        case -58:
            return "CL_INVALID_EVENT";
        case -59:
            return "CL_INVALID_OPERATION";
        case -60:
            return "CL_INVALID_GL_OBJECT";
        case -61:
            return "CL_INVALID_BUFFER_SIZE";
        case -62:
            return "CL_INVALID_MIP_LEVEL";
        case -63:
            return "CL_INVALID_GLOBAL_WORK_SIZE";
        case -64:
            return "CL_INVALID_PROPERTY";
        case -65:
            return "CL_INVALID_IMAGE_DESCRIPTOR";
        case -66:
            return "CL_INVALID_COMPILER_OPTIONS";
        case -67:
            return "CL_INVALID_LINKER_OPTIONS";
        case -68:
            return "CL_INVALID_DEVICE_PARTITION_COUNT";

        // extension errors
        case -1000:
            return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
        case -1001:
            return "CL_PLATFORM_NOT_FOUND_KHR";
        case -1002:
            return "CL_INVALID_D3D10_DEVICE_KHR";
        case -1003:
            return "CL_INVALID_D3D10_RESOURCE_KHR";
        case -1004:
            return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
        case -1005:
            return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
        default:
            return "Unknown OpenCL error";
    }
}

/* Find a GPU or CPU associated with the first available platform */
static cl_device_id create_device() {
    cl_platform_id platform;
    cl_device_id dev;
    int err;

    /* Identify a platform */
    err = clGetPlatformIDs(1, &platform, NULL);
    ERROR("Couldn't identify a platform");

    /* Access a device */
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
    if (err == CL_DEVICE_NOT_FOUND)
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &dev, NULL);
    ERROR("Couldn't access any devices");

    return dev;
}

/* Create program from a file and compile it */
static cl_program build_program(cl_context ctx, cl_device_id dev) {
    cl_program program;
    FILE* program_handle;
    char *program_buffer, *program_log;
    size_t program_size, log_size;
    cl_int err = 0;

    /* Read program file and place content into buffer */
    program_handle = fopen(PROGRAM_FILE, "r");
    if (program_handle == NULL) {
        perror("Couldn't find the program cl file");
        exit(1);
    }

    fseek(program_handle, 0, SEEK_END);
    program_size = ftell(program_handle);
    rewind(program_handle);
    program_buffer = (char*)malloc(program_size + 1);
    program_buffer[program_size] = '\0';
    fread(program_buffer, sizeof(char), program_size, program_handle);
    fclose(program_handle);

    /* Create program from file */
    program = clCreateProgramWithSource(ctx, 1,
                                        (const char**)&program_buffer, &program_size, &err);
    ERROR("Couldn't create the program");

    //free(program_buffer);
    clSVMFree(ctx, program_buffer);

    /* Build program */
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err < 0) {
        /* Find size of log and print to std output */
        clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG,
                              0, NULL, &log_size);
        program_log = (char*)malloc(log_size + 1);
        program_log[log_size] = '\0';
        clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG,
                              log_size + 1, program_log, NULL);
        printf("%s\n", program_log);

        //free(program_log);
        clSVMFree(ctx, program_log);

        exit(1);
    }

    return program;
}

//void init_opencl(void) {
void init_opencl(void) {
    /* Create device and context */
    device = create_device();
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    ERROR("Couldn't create a context");

    /* Build program */
    program = build_program(context, device);

    /* Create a command queue */
    queue = clCreateCommandQueue(context, device, 0, &err);
    ERROR("Couldn't create a command queue");
}

void free_opencl(void) {
    clReleaseCommandQueue(queue);
    clReleaseProgram(program);
    clReleaseContext(context);
    clReleaseDevice(device);
}

void create_dataset(linear_param_t* params, data_t* dataset) {
    FILE* ptr_file = fopen(params->filename, "r");
    if (ptr_file == NULL) {
        perror("Failed to load dataset file");
        exit(1);
    }
    //dataset = (data_t *)clSVMAlloc(context, CL_MEM_READ_WRITE, sizeof(data_t) * params->size, 0);

    char* token;
    char buf[1024];

    for (int i = 0; i < int(params->size) && fgets(buf, 1024, ptr_file) != NULL; i++) {
        token = strtok(buf, "\t");
        dataset[i].x = atof(token);
        token = strtok(NULL, "\t");
        dataset[i].y = atof(token);
        //printf("Dataset: %f, %f\n", dataset[i].x, dataset[i].y);
    }
    fclose(ptr_file);
}

void temperature_regression(results_t* results) {
    linear_param_t params;
    params.filename = TEMP_FILENAME;
    params.size = TEMP_SIZE;
    params.wg_size = TEMP_WORKGROUP_SIZE;
    params.wg_count = TEMP_WORKGROUP_NBR;

    data_t* dataset = (data_t*)clSVMAlloc(context, CL_MEM_READ_WRITE, sizeof(data_t) * params.size, 0);
    create_dataset(&params, dataset);

    parallelized_regression(&params, dataset, &results->parallelized);
    iterative_regression(&params, dataset, &results->iterative);
    clSVMFree(context, dataset);
}

void r_squared(
    linear_param_t* params,
    data_t* dataset,
    sum_t* linreg,
    result_t* response) {
    //cl_mem  dataset_buffer, result_buffer;
    cl_kernel kernel;

    //rsquared_t * results = NULL;
    cl_float mean = linreg->sumy / params->size;
    cl_float equation[2] = {response->a0, response->a1};

    //results = (rsquared_t*) malloc(sizeof(rsquared_t) * params->wg_count);

    /* Create a kernel */
    kernel = clCreateKernel(program, KERNEL_RSQUARED_FUNC, &err);
    ERROR("Couldn't create a kernel");

    /* Max workgroup size for kernel */
    size_t max_wg_size = 0;
    err = clGetKernelWorkGroupInfo(kernel, device,
                                   CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &max_wg_size, NULL);
    ERROR("Couldn't read kernel work group info");
    if (params->wg_size > max_wg_size) {
        perror("Workgroup size kernel too high");
        exit(1);
    }


    rsquared_t* result_buffer = (rsquared_t*)clSVMAlloc(context, CL_MEM_READ_WRITE, params->wg_count * sizeof(rsquared_t), 0);

    size_t globalWorkSize[1];
    globalWorkSize[0] = params->size;
    if (params->size % params->wg_size)
        globalWorkSize[0] += params->wg_size - (params->size % params->wg_size);

    size_t global_offset[2] = {0, 1};
    bool gpu_run = true, cpu_run = false;

    size_t cpu_global_size[3];
    cpu_global_size[0] = cpu_offset * globalWorkSize[0] / 100;

    if (cpu_global_size[0] % params->wg_size != 0) {
        cpu_global_size[0] = (1 + cpu_global_size[0] / params->wg_size) * params->wg_size;
    }
    cpu_global_size[1] = cpu_global_size[2] = 1;

    size_t gpu_global_size[3];
    gpu_global_size[0] = globalWorkSize[0] - cpu_global_size[0];
    if (gpu_global_size[0] <= 0) {
        gpu_run = false;
    }
    gpu_global_size[1] = gpu_global_size[2] = 1;
    global_offset[0] = cpu_global_size[0];

    if (cpu_offset > 0) {
        cpu_run = true;
    }


    cl_event event1;

    if (gpu_run) {
        err = clSetKernelArgSVMPointer(kernel, 0, (void*)dataset);
        err |= clSetKernelArg(kernel, 1, sizeof(cl_float), (void*)&mean);
        err |= clSetKernelArg(kernel, 2, sizeof(equation), (void*)&equation);
        err |= clSetKernelArg(kernel, 3, params->wg_size * sizeof(rsquared_t), NULL);
        err |= clSetKernelArgSVMPointer(kernel, 4, (void*)(result_buffer + (cpu_global_size[0] / local_size)));
        err = clEnqueueNDRangeKernel(queue, kernel, 1, global_offset, gpu_global_size,
                                     &params->wg_size, 0, NULL, &event1);
        if (err != CL_SUCCESS)
            printf("ERROR1 in corun\n");
    }
    if (cpu_run) {
        rsquaredOMP(dataset, mean, equation, result_buffer, cpu_global_size[0]);
    }

    if (gpu_run) {
        err = clWaitForEvents(1, &event1);
        if (err != CL_SUCCESS)
            printf("ERROR in corun\n");
    }

    ERROR("Couldn't create a kernel argument");

    START_TIME
    /* Enqueue kernel */

    rsquared_t final = {0};
    rsquared_t* results = (rsquared_t*)result_buffer;

    for (int i = 0; i < int(params->wg_count); i++) {
        final.actual += results[i].actual;
        final.estimated += results[i].estimated;
    }

    response->rsquared = final.estimated / final.actual * 100;

    END_TIME
    response->time += CALC_TIME;

    //free(results);

    clReleaseKernel(kernel);
    clSVMFree(context, result_buffer);
}

void parallelized_regression(
    linear_param_t* params,
    data_t* dataset,
    result_t* response) {
    /* Data and buffers */
    //sum_t * results;
    cl_kernel kernel;

    sum_t* result_buffer = (sum_t*)clSVMAlloc(context, CL_MEM_READ_WRITE, params->wg_count * sizeof(sum_t), 0);
    ERROR("Couldn't create a buffer");

    /* Create a kernel */
    kernel = clCreateKernel(program, KERNEL_LINREG_FUNC, &err);
    ERROR("Couldn't create a kernel");

    /* Max workgroup size for kernel */
    size_t max_wg_size = 0;
    err = clGetKernelWorkGroupInfo(kernel, device,
                                   CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &max_wg_size, NULL);
    ERROR("Couldn't read kernel work group info");
    if (params->wg_size > max_wg_size) {
        perror("Workgroup size kernel too high");
        exit(1);
    }

    size_t globalWorkSize[1];
    globalWorkSize[0] = params->size;
    if (params->size % params->wg_size)
        globalWorkSize[0] += params->wg_size - (params->size % params->wg_size);

    size_t global_offset[2] = {0, 1};
    bool gpu_run = true, cpu_run = false;

    size_t cpu_global_size[3];
    cpu_global_size[0] = cpu_offset * globalWorkSize[0] / 100;
    if (cpu_global_size[0] % params->wg_size != 0) {
        cpu_global_size[0] = (1 + cpu_global_size[0] / params->wg_size) * params->wg_size;
    }
    cpu_global_size[1] = cpu_global_size[2] = 1;

    size_t gpu_global_size[3];
    gpu_global_size[0] = globalWorkSize[0] - cpu_global_size[0];
    if (gpu_global_size[0] <= 0) {
        gpu_run = false;
    }
    gpu_global_size[1] = gpu_global_size[2] = 1;
    global_offset[0] = cpu_global_size[0];

    if (cpu_offset > 0) {
        cpu_run = true;
    }


    cl_event event2;

    if (gpu_run) {
        err = clSetKernelArgSVMPointer(kernel, 0, (void*)dataset);
        err |= clSetKernelArg(kernel, 1, params->wg_size * sizeof(sum_t), NULL);
        err |= clSetKernelArgSVMPointer(kernel, 2, (void*)(result_buffer + (cpu_global_size[0] / local_size)));
        err = clEnqueueNDRangeKernel(queue, kernel, 1, global_offset, gpu_global_size,
                                     &params->wg_size, 0, NULL, &event2);
        if (err != CL_SUCCESS)
            printf("ERROR2 in corun\n");
    }
    if (cpu_run) {
        linear_regressionOMP(dataset, result_buffer, cpu_global_size[0]);
    }
    if (gpu_run) {
        err = clWaitForEvents(1, &event2);
        if (err != CL_SUCCESS)
            printf("ERROR in corun\n");
    }

    /* Start timer */
    START_TIME

    /* Enqueue kernel */

    ERROR("Couldn't enqueue the kernel");


    /* Finalize algorithm */
    sum_t final = {0};
    sum_t* results = (sum_t*)result_buffer;  //

    for (int i = 0; i < int(params->wg_count); i++) {
        final.sumx += results[i].sumx;
        final.sumy += results[i].sumy;
        final.sumxy += results[i].sumxy;
        final.sumxsq += results[i].sumxsq;
    }
    double denom = (params->size * final.sumxsq - (final.sumx * final.sumx));
    response->a0 = (final.sumy * final.sumxsq - final.sumx * final.sumxy) / denom;
    response->a1 = (params->size * final.sumxy - final.sumx * final.sumy) / denom;

    /* End timer */
    END_TIME
    response->time = CALC_TIME;

    /* Deallocate resources  */
    clReleaseKernel(kernel);
    clSVMFree(context, result_buffer);
    r_squared(params, dataset, &final, response);
}
