// includes, system
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include "backprop.h"

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS


#ifdef NV  //NVIDIA
#include <oclUtils.h>
#else
#include <CL/cl.h>
#endif

////////////////////////////////////////////////////////////////////////////////

// local variables
static cl_context context;
static cl_command_queue cmd_queue;
static cl_device_type device_type;
static cl_device_id* device_list;
static cl_int num_devices;

extern int cpu_offset;

extern double gettime();

#define THREADS 256
#define WIDTH 16
#define HEIGHT 16

#define WM(i, j) weight_matrix[(j) + (i)*WIDTH]


void bpnn_layerforward_omp(float* input_cuda,
                           float* output_hidden_cuda,
                           float* input_hidden_cuda,
                           float* hidden_partial_sum,
                           float* input_node,
                           float* weight_matrix,
                           int in,
                           int hid,
                           int size) {
    input_node = (float*)malloc(sizeof(float) * HEIGHT);
    weight_matrix = (float*)malloc(sizeof(float) * HEIGHT * WIDTH);

    // #pragma omp parallel for
    for (int by = 0; by < size; by++) {
        for (int tx = 0; tx < 16; tx++) {
            for (int ty = 0; ty < 16; ty++) {
                int index = (hid + 1) * HEIGHT * by + (hid + 1) * ty + tx + 1 + (hid + 1);
                int index_in = HEIGHT * by + ty + 1;

                if (tx == 0)
                    input_node[ty] = input_cuda[index_in];
                //#pragma omp barrier

                weight_matrix[ty * WIDTH + tx] = weight_matrix[ty * WIDTH + tx] * input_node[ty];
                //#pragma omp barrier

                for (int i = 1; i <= HEIGHT; i = i * 2) {
                    int power_two = i;
                    if (ty % power_two == 0) {
                        weight_matrix[ty * WIDTH + tx] = weight_matrix[ty * WIDTH + tx] + weight_matrix[(ty + power_two / 2) * WIDTH + tx];
                        //#pragma omp barrier
                    }
                }
                // change
                input_hidden_cuda[index] = weight_matrix[ty * WIDTH + tx];
                //#pragma omp barrier

                if (tx == 0) {
                    // BUG?
                    hidden_partial_sum[by * hid + ty] = weight_matrix[tx * WIDTH + ty];
                }
            }
        }
    }
}

void bpnn_adjust_weights_omp(float* delta,
                             int hid,
                             float* ly,
                             int in,
                             float* w,
                             float* oldw,
                             int size) {
#pragma omp parallel for
    for (int by = 0; by < size; by++) {
        for (int tx = 0; tx < 16; tx++) {
            for (int ty = 0; ty < 16; ty++) {
                int index = (hid + 1) * HEIGHT * by + (hid + 1) * ty + tx + 1 + (hid + 1);
                int index_y = HEIGHT * by + ty + 1;
                int index_x = tx + 1;

                w[index] += ((ETA * delta[index_x] * ly[index_y]) + (MOMENTUM * oldw[index]));
                oldw[index] = ((ETA * delta[index_x] * ly[index_y]) + (MOMENTUM * oldw[index]));
#pragma omp barrier
                if (ty == 0 && by == 0) {
                    w[index_x] += ((ETA * delta[index_x]) + (MOMENTUM * oldw[index_x]));
                    oldw[index_x] = ((ETA * delta[index_x]) + (MOMENTUM * oldw[index_x]));
                }
            }
        }
    }
}

static int initialize(int use_gpu) {
    cl_int result;
    size_t size;

    // create OpenCL context
    cl_platform_id platform_id;
    if (clGetPlatformIDs(1, &platform_id, NULL) != CL_SUCCESS) {
        printf("ERROR: clGetPlatformIDs(1,*,0) failed\n");
        return -1;
    }
    cl_context_properties ctxprop[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)platform_id, 0};
    device_type = use_gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU;
    context = clCreateContextFromType(ctxprop, device_type, NULL, NULL, NULL);
    if (!context) {
        printf("ERROR: clCreateContextFromType(%s) failed\n", use_gpu ? "GPU" : "CPU");
        return -1;
    }

    // get the list of GPUs
    result = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &size);
    num_devices = (int)(size / sizeof(cl_device_id));
    // printf("num_devices = %d\n", num_devices);

    if (result != CL_SUCCESS || num_devices < 1) {
        printf("ERROR: clGetContextInfo() failed\n");
        return -1;
    }
    device_list = new cl_device_id[num_devices];
    //device_list = (cl_device_id *)malloc(sizeof(cl_device_id)*num_devices);
    if (!device_list) {
        printf("ERROR: new cl_device_id[] failed\n");
        return -1;
    }
    result = clGetContextInfo(context, CL_CONTEXT_DEVICES, size, device_list, NULL);
    if (result != CL_SUCCESS) {
        printf("ERROR: clGetContextInfo() failed\n");
        return -1;
    }

    // create command queue for the first device
    cmd_queue = clCreateCommandQueue(context, device_list[0], 0, NULL);
    if (!cmd_queue) {
        printf("ERROR: clCreateCommandQueue() failed\n");
        return -1;
    }
    return 0;
}

static int shutdown() {
    // release resources
    if (cmd_queue)
        clReleaseCommandQueue(cmd_queue);
    if (context)
        clReleaseContext(context);
    if (device_list)
        delete[] device_list;

    // reset all variables
    cmd_queue = 0;
    context = 0;
    device_list = 0;
    num_devices = 0;
    device_type = 0;

    return 0;
}

unsigned int num_threads = 0;
unsigned int num_blocks = 0;

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {
    setup(argc, argv);
    return 0;
}

int bpnn_train_kernel(BPNN* net, float* eo, float* eh) {
    int in, hid, out;
    float out_err, hid_err;

    in = net->input_n;
    hid = net->hidden_n;
    out = net->output_n;

    int sourcesize = 1024 * 1024;
    char* source = (char*)calloc(sourcesize, sizeof(char));
    if (!source) {
        printf("ERROR: calloc(%d) failed\n", sourcesize);
        return -1;
    }

    // read the kernel core source
    char kernel_bp1[] = "bpnn_layerforward_ocl";
    char kernel_bp2[] = "bpnn_adjust_weights_ocl";
    char tempchar[] = "./backprop_kernel.cl";
    FILE* fp = fopen(tempchar, "rb");
    if (!fp) {
        printf("ERROR: unable to open '%s'\n", tempchar);
        return -1;
    }
    size_t tmp_s = fread(source + strlen(source), sourcesize, 1, fp);
    fclose(fp);

    int use_gpu = 1;
    if (initialize(use_gpu))
        return -1;

    // compile kernel
    cl_int err = 0;
    const char* slist[2] = {source, 0};
    cl_program prog = clCreateProgramWithSource(context, 1, slist, NULL, &err);
    if (err != CL_SUCCESS) {
        printf("ERROR: clCreateProgramWithSource() => %d\n", err);
        return -1;
    }
    err = clBuildProgram(prog, 0, NULL, NULL, NULL, NULL);

    if (err != CL_SUCCESS) {
        printf("ERROR: clBuildProgram() => %d\n", err);
        return -1;
    }

    cl_kernel kernel1;
    cl_kernel kernel2;
    kernel1 = clCreateKernel(prog, kernel_bp1, &err);
    kernel2 = clCreateKernel(prog, kernel_bp2, &err);
    if (err != CL_SUCCESS) {
        printf("ERROR: clCreateKernel() 0 => %d\n", err);
        return -1;
    }
    clReleaseProgram(prog);

    float* input_weights_one_dim;
    float* input_weights_prev_one_dim;
    //float * partial_sum;
    float sum;
    int num_blocks = in / BLOCK_SIZE;

    input_weights_one_dim = (float*)clSVMAlloc(context, CL_MEM_READ_WRITE, (in + 1) * (hid + 1) * sizeof(float), 0);
    if (input_weights_one_dim == NULL) {
        printf("ERROR: clSVMAlloc input_weights_one_dim\n");
        return -1;
    }
    input_weights_prev_one_dim = (float*)clSVMAlloc(context, CL_MEM_READ_WRITE, (in + 1) * (hid + 1) * sizeof(float), 0);
    if (input_weights_prev_one_dim == NULL) {
        printf("ERROR: clSVMAlloc input_weights_prev_one_dim\n");
        return -1;
    }

    //partial_sum = (float *) malloc(num_blocks * WIDTH * sizeof(float));

    // set global and local workitems
    size_t global_work[3] = {BLOCK_SIZE, size_t(BLOCK_SIZE * num_blocks), 1};
    size_t local_work[3] = {BLOCK_SIZE, BLOCK_SIZE, 1};

    // this preprocessing stage is temporarily added to correct the bug of wrong memcopy using two-dimensional net->inputweights
    // todo: fix mem allocation
    int m = 0;
    for (int k = 0; k <= in; k++) {
        for (int j = 0; j <= hid; j++) {
            input_weights_one_dim[m] = net->input_weights[k][j];
            input_weights_prev_one_dim[m] = net->input_prev_weights[k][j];
            m++;
        }
    }

    void* input_hidden_ocl;
    void* input_ocl;
    void* output_hidden_ocl;
    float* hidden_partial_sum;
    void* hidden_delta_ocl;
    void* input_prev_weights_ocl;

    input_hidden_ocl = input_weights_one_dim;

    input_ocl = clSVMAlloc(context, CL_MEM_READ_WRITE, (in + 1) * sizeof(float), 0);
    if (input_ocl == NULL) {
        printf("ERROR: clSVMAlloc input_ocl\n");
        return -1;
    }
    output_hidden_ocl = clSVMAlloc(context, CL_MEM_READ_WRITE, (hid + 1) * sizeof(float), 0);
    if (output_hidden_ocl == NULL) {
        printf("ERROR: clSVMAlloc output_hidden_ocl\n");
        return -1;
    }
    hidden_partial_sum = (float*)clSVMAlloc(context, CL_MEM_READ_WRITE, num_blocks * WIDTH * sizeof(float), 0);
    if (hidden_partial_sum == NULL) {
        printf("ERROR: clSVMAlloc hidden_partial_sum\n");
        return -1;
    }
    hidden_delta_ocl = clSVMAlloc(context, CL_MEM_READ_WRITE, (hid + 1) * sizeof(float), 0);
    if (hidden_delta_ocl == NULL) {
        printf("ERROR: clSVMAlloc hidden_delta_ocl\n");
        return -1;
    }

    input_prev_weights_ocl = input_weights_prev_one_dim;

    memcpy(input_ocl, net->input_units, (in + 1) * sizeof(float));

    bool gpu_run = true, cpu_run = false;
    int work_dim = 1;
    int cpu_num_blocks = cpu_offset * num_blocks / 100;
    size_t cpu_global_size[3] = {BLOCK_SIZE, size_t(BLOCK_SIZE * cpu_num_blocks), 1};
    size_t gpu_global_size[3] = {BLOCK_SIZE, size_t(BLOCK_SIZE * (num_blocks - cpu_num_blocks)), 1};
    size_t global_offset[2] = {0, size_t(BLOCK_SIZE * cpu_num_blocks)};

    if (cpu_offset > 0) {
        cpu_run = true;
    }
    //size_t global_work[3] = {BLOCK_SIZE, BLOCK_SIZE * num_blocks, 1};

    if (gpu_global_size[1] <= 0) {
        gpu_run = false;
    }
    // printf("CPU size: %ld, GPU size: %ld\n", cpu_global_size[1], gpu_global_size[1]);

    cl_event kernelEvent1;

    double tstart = gettime();

    if (gpu_run) {
        // printf("GPU running\n");
        clSetKernelArgSVMPointer(kernel1, 0, (void*)input_ocl);
        clSetKernelArgSVMPointer(kernel1, 1, (void*)output_hidden_ocl);
        clSetKernelArgSVMPointer(kernel1, 2, (void*)input_hidden_ocl);
        clSetKernelArgSVMPointer(kernel1, 3, (void*)hidden_partial_sum);
        clSetKernelArg(kernel1, 4, sizeof(float) * HEIGHT, (void*)NULL);
        clSetKernelArg(kernel1, 5, sizeof(float) * HEIGHT * WIDTH, (void*)NULL);
        clSetKernelArg(kernel1, 6, sizeof(cl_int), (void*)&in);
        clSetKernelArg(kernel1, 7, sizeof(cl_int), (void*)&hid);
        err = clEnqueueNDRangeKernel(cmd_queue, kernel1, 2, global_offset, gpu_global_size, local_work, 0, 0, &kernelEvent1);
        if (err != CL_SUCCESS)
            printf("ERROR1 in corun\n");
    }
    if (cpu_run) {
        // printf("CPU running\n");
        bpnn_layerforward_omp((float*)input_ocl, (float*)output_hidden_ocl, (float*)input_hidden_ocl, (float*)hidden_partial_sum, NULL, NULL, in, hid, cpu_num_blocks);
    }
    if (gpu_run) {
        cl_int err = clWaitForEvents(1, &kernelEvent1);
        if (err != CL_SUCCESS)
            printf("ERROR in corun\n");
    }

    for (int j = 1; j <= hid; j++) {
        sum = 0.0;
        for (int k = 0; k < num_blocks; k++) {
            sum += hidden_partial_sum[k * hid + j - 1];
        }
        sum += net->input_weights[0][j];
        net->hidden_units[j] = float(1.0 / (1.0 + exp(-sum)));
    }

    bpnn_layerforward(net->hidden_units, net->output_units, net->hidden_weights, hid, out);
    bpnn_output_error(net->output_delta, net->target, net->output_units, out, &out_err);
    bpnn_hidden_error(net->hidden_delta, hid, net->output_delta, out, net->hidden_weights, net->hidden_units, &hid_err);
    bpnn_adjust_weights(net->output_delta, out, net->hidden_units, hid, net->hidden_weights, net->hidden_prev_weights);

    memcpy(hidden_delta_ocl, net->hidden_delta, (hid + 1) * sizeof(float));
    if (gpu_run) {
        clSetKernelArgSVMPointer(kernel2, 0, (void*)hidden_delta_ocl);
        clSetKernelArg(kernel2, 1, sizeof(cl_int), (void*)&hid);
        clSetKernelArgSVMPointer(kernel2, 2, (void*)input_ocl);
        clSetKernelArg(kernel2, 3, sizeof(cl_int), (void*)&in);
        clSetKernelArgSVMPointer(kernel2, 4, (void*)input_hidden_ocl);
        clSetKernelArgSVMPointer(kernel2, 5, (void*)input_prev_weights_ocl);
        err = clEnqueueNDRangeKernel(cmd_queue, kernel2, 2, global_offset, gpu_global_size, local_work, 0, 0, &kernelEvent1);
        if (err != CL_SUCCESS) {
            printf("ERROR: 1  clEnqueueNDRangeKernel()=>%d failed\n", err);
            return -1;
        }
    }
    if (cpu_run) {
        bpnn_adjust_weights_omp((float*)hidden_delta_ocl, hid, (float*)input_ocl, in, (float*)input_hidden_ocl, (float*)input_prev_weights_ocl, cpu_num_blocks);
    }
    if (gpu_run) {
        cl_int err = clWaitForEvents(1, &kernelEvent1);
        if (err != CL_SUCCESS)
            printf("ERROR in corun\n");
    }

    memcpy(net->input_units, input_ocl, (in + 1) * sizeof(float));

    clSVMFree(context, input_hidden_ocl);
    clSVMFree(context, input_ocl);
    clSVMFree(context, output_hidden_ocl);
    clSVMFree(context, hidden_partial_sum);
    clSVMFree(context, hidden_delta_ocl);
    clSVMFree(context, input_prev_weights_ocl);
    double tend = gettime();
    printf("Total time: %lf ms\n\n", 1000.0 * (tend - tstart));

}
