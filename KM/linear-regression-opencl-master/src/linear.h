#ifndef LINEAR_H__
#define LINEAR_H__

#include <time.h>

#ifdef MAC
  #include <OpenCL/cl.h>
#else
  #include <CL/cl.h>
#endif

#define PROGRAM_FILE "src/linear.cl"
#define KERNEL_LINREG_FUNC "linear_regression"
#define KERNEL_RSQUARED_FUNC "rsquared"

#define RESULT_FILENAME "assets/_results.txt"

#define TEMP_FILENAME "assets/temperature.txt"
#define TEMP_SIZE 96453
#define TEMP_WORKGROUP_SIZE 63
#define TEMP_WORKGROUP_NBR TEMP_SIZE / TEMP_WORKGROUP_SIZE // 1531

#define HOUSE_FILENAME "assets/house.txt"
#define HOUSE_SIZE 545
#define HOUSE_WORKGROUP_SIZE 109
#define HOUSE_WORKGROUP_NBR HOUSE_SIZE / HOUSE_WORKGROUP_SIZE // 5

#define CALC_TIME (((double) (end - start)) / CLOCKS_PER_SEC) * 1000
#define START_TIME start = clock();
#define END_TIME end = clock();

#define LOG_DATASET() \
  for (int i = 0; i < DATASET_SIZE; i++) \
    printf("(%f, %f)\n", dataset[i].x, dataset[i].y);

#define LOG_DATA_T(var) printf("data_t:\n\tx: %f\n\ty: %f\n", var.x, var.y);
#define LOG_SUM_T(var) printf("sum_t:\n\tsumx: %f\n\tsumy: %f\n\tsumxy: %f\n\tsumxsq: %f\n", var.sumx, var.sumy, var.sumxy, var.sumxsq);
#define LOG_RESULT_T(var) printf("result_t:\n\ta0: %f\n\ta1: %f\n\ttime: %f\n", var.a0, var.a1, var.time);
#define LOG_RSQUARED_T(var) printf("rsquared_t:\n\tactual: %f\n\testimated: %f\n", var.actual, var.estimated)

#define ERROR(msg) \
  if(err < 0) { \
    perror(msg); \
    fprintf(stderr, "OpenCL error : %s\n", error_msg(err)); \
    exit(1); \
  }

/* a0 # a1 # time # rsquared */
#define WRITE_RESULT(file, result) \
  fprintf(file, "%.3f#%.3f#%.3f#%d\n", \
    result.a0, \
    result.a1, \
    result.time, \
    result.rsquared);

#define PRINT_RESULT(title, result) \
  printf("\t%s\n\t--------\n\t| R squared: %d%%\n\t| Time: %.3fms\n\t| Equation: y = %.3fx + %.3f\n\n", \
    title, \
    result.rsquared, \
    result.time, \
    result.a1, \
    result.a0);

typedef struct {
 char * filename;
 size_t size;
 size_t wg_size;
 size_t wg_count;
} linear_param_t;

typedef struct {
  cl_float x;
  cl_float y;
} data_t;

typedef struct {
  cl_float sumx;
  cl_float sumy;
  cl_float sumxy;
  cl_float sumxsq;
} sum_t;

typedef struct {
  cl_float a0;
  cl_float a1;
  int rsquared;
  double time;
} result_t;

typedef struct {
  result_t iterative;
  result_t parallelized;
} results_t;

typedef struct {
  cl_float actual;
  cl_float estimated;
} rsquared_t;

extern clock_t start, end;

void init_opencl(void);
void free_opencl(void);
void parallelized_regression(linear_param_t *, data_t *, result_t *);
void iterative_regression(linear_param_t *, data_t *, result_t *);

#endif
