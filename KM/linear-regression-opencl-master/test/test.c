#include <stdio.h>
#include <time.h>
#include <OpenCL/cl.h>

cl_platform_id platform;
cl_device_id device;
cl_context context;
cl_program program;
cl_command_queue queue;
cl_kernel kernel;

void create_program(char const *cl_file);

void numeric_reduction() {
  create_program("numeric_reduction.cl");

  const size_t glob_size = 1024;
  int data[glob_size];

  for (int i = 0 ; i < glob_size ; i++)
    data[i] = i;

  size_t wg_size = 0;
  kernel = clCreateKernel(program, "numeric_reduction", NULL);
  clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &wg_size, NULL);

  const size_t num_wg = glob_size / wg_size;
  int result[num_wg];

  cl_mem buf_in = clCreateBuffer(context, 
    CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR,
    sizeof(int) * glob_size, data, NULL);

  cl_mem buf_out = clCreateBuffer(context, 
    CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY,
    sizeof(int) * num_wg, NULL, NULL);

  kernel = clCreateKernel(program, "numeric_reduction", NULL);
  clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf_in);
  clSetKernelArg(kernel, 1, sizeof(int) * wg_size, NULL);
  clSetKernelArg(kernel, 2, sizeof(cl_mem), &buf_out);

  queue = clCreateCommandQueue(context, device, 0, NULL);
  clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &glob_size, &wg_size, 0, NULL, NULL);
  clEnqueueReadBuffer(queue, buf_out, CL_TRUE, 0, sizeof(int) * num_wg, result, 0, NULL, NULL);

  for (int i = 0; i < num_wg; i++)
    printf("%d,", result[i]);
  printf("=%d\n", result[0] + result[1]);
}

void array_2d() {
  create_program("array_2d.cl");

  size_t const x = 2, y = 3;
  size_t dim[] = { x, y }; 
  int arr[y][x] = { 
    {10, 11}, 
    {20, 21}, 
    {30, 31} 
  };

  cl_mem mem_arr = clCreateBuffer(
    context, 
    CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
    sizeof(int) * x * y, 
    arr, NULL);

  kernel = clCreateKernel(program, "array_2d", NULL);
  clSetKernelArg(kernel, 0, sizeof(cl_mem), &mem_arr);

  queue = clCreateCommandQueue(context, device, 0, NULL);
  clEnqueueNDRangeKernel(queue, kernel, 2, NULL, dim, NULL, 0, NULL, NULL);
  clEnqueueReadBuffer(queue, mem_arr, CL_TRUE, 0, sizeof(int) * x * y, arr, 0, NULL, NULL);

  for (int i = 0; i < y; i++)
    for (int j = 0; j < x; j++)
      printf("%d\n", arr[i][j]);
}

void array_st() {
  create_program("array_st.cl");
  size_t glob_size = 100;

  struct st {
    int x, y;
  };

  struct st data[glob_size];

  for (int i = 0; i < glob_size; i++) {
    data[i].x = i;
    data[i].y = i + 1;
  }

  cl_int err = 0;
  cl_mem buf = clCreateBuffer(context, 
    CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
    sizeof(struct st) * glob_size, data, &err);

  kernel = clCreateKernel(program, "array_st", &err);
  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf);

  queue = clCreateCommandQueue(context, device, 0, &err);
  err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &glob_size, NULL, 0, NULL, NULL);
  
  clEnqueueReadBuffer(queue, buf, CL_TRUE, 0, sizeof(struct st) * glob_size, data, 0, NULL, NULL);
  printf("(%d,%d)", data[0].x, data[0].y);
}

void array() {
  create_program("array.cl");
  size_t glob_size = 100;
  size_t loc_size = 1; 
  int data[glob_size];
  int out_data[glob_size];

  for (int i = 0 ; i < glob_size ; data[i++] = i + 1)
    ;

  cl_mem mem_data = clCreateBuffer(
    context, 
    CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, 
    sizeof(int) * glob_size, 
    data, NULL);

  cl_mem mem_out_data = clCreateBuffer(
    context, 
    CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, 
    sizeof(int) * glob_size, 
    NULL, NULL);
  
  // clEnqueueFillBuffer(queue, mem_data, )

  // Kernel
  kernel = clCreateKernel(program, "array", NULL);
  clSetKernelArg(kernel, 0, sizeof(cl_mem), &mem_data);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), &mem_out_data);

  queue = clCreateCommandQueue(context, device, 0, NULL);
  clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &glob_size, NULL, 0, NULL, NULL);

  clock_t start, end;
  double cpu_time_used;
  
  start = clock();
  clEnqueueReadBuffer(queue, mem_out_data, CL_TRUE, 0, sizeof(int) * glob_size, out_data, 0, NULL, NULL);
  //clEnqueueMapBuffer(queue, mem_out_data, CL_TRUE, CL_MAP_READ)

  // clFinish(queue); // wait queue end
  end = clock();
  cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
  
  printf("\ntime: %f seconds\n", cpu_time_used);

  // for (int i = 0; i < glob_size; i++)
    printf("%d\n", out_data[0]);

  // 512: 0.000491 seconds
  // 10:  0.001737 seconds 
}

void hello() {
  create_program("hello.cl");
  // data passed to kernel
  char hello_msg[10];
  cl_mem mem = clCreateBuffer(context, 
    CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, 
    sizeof(hello_msg), 
    NULL, NULL);
  
  // Kernel
  kernel = clCreateKernel(program, "hello", NULL);
  clSetKernelArg(kernel, 0, sizeof(cl_mem), &mem);

  // Send command
  queue = clCreateCommandQueue(context, device, 0, NULL);
  clEnqueueTask(queue, kernel, 0, NULL, NULL); // Execute the kernel ones !
  clEnqueueReadBuffer(queue, mem, CL_TRUE, 0, sizeof(hello_msg), hello_msg, 0, NULL, NULL);

  printf("%s", hello_msg);
}

void device_info() {
  cl_platform_id platform;
  cl_device_id devices[2];
  
  cl_uint num_platform, num_device;

  clGetPlatformIDs(1, &platform, &num_platform);
  clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, devices, &num_device);

  char vers[512];
  cl_uint maxcu;
  size_t x;
  clGetDeviceInfo(devices[0], CL_DEVICE_NAME, sizeof(vers), &vers, 0);
  printf("%s\n", vers);
  clGetDeviceInfo(devices[0], CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &x, 0);
  printf("%u\n", x);
}

void create_program(char const *cl_file) {
  clGetPlatformIDs(1, &platform, NULL);
  clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
  context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
  
  FILE *program_file = fopen(cl_file, "r");
  char* program_buffer = NULL;
  size_t program_size;

  fseek(program_file, 0, SEEK_END); // go to end of file
  program_size = ftell(program_file); // get position indicator of end of file so its size
  rewind(program_file); // go to beginning of file
  program_buffer = (char*) malloc(program_size + 1); // allocates memory of program
  program_buffer[program_size] = '\0'; // set null char at end of buffer
  fread(program_buffer, sizeof(char), program_size, program_file); // copy file content in buffer
  fclose(program_file);

  // create program
  program = clCreateProgramWithSource(context, 1, (const char**) &program_buffer, &program_size, NULL);
  free(program_buffer);
  
  // build program
  clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
}

int main() {
  // array_st();
  printf("%d\n", sizeof(cl_int));
  printf("%d\n", sizeof(cl_int4));

  cl_int4 x = {1, 2, 3, 4};
  // printf("%d\n", sizeof(cl_int));
}
