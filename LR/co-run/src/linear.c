#include <stdio.h>
#include <string.h>
#include "linear.h"
#include <sys/time.h>


double gettime() {
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec + t.tv_usec * 1e-6;
}


clock_t start;
clock_t end;

extern int cpu_offset;

/* Read file */
static
void create_dataset(linear_param_t * params, data_t * dataset) {
  FILE *ptr_file = fopen(params->filename, "r");
  if (ptr_file == NULL) {
    perror("Failed to load dataset file");
    exit(1);
  }

  char *token;
  char buf[1024];

  for (int i = 0; i < params->size && fgets(buf, 1024, ptr_file) != NULL; i++) {
    token = strtok(buf, "\t");
    dataset[i].x = atof(token);
    token = strtok(NULL, "\t");
    dataset[i].y = atof(token);
  }

  fclose(ptr_file);
}

// static
// void house_regression(results_t * results) {
//   linear_param_t params;
//   params.filename = HOUSE_FILENAME;
//   params.size = HOUSE_SIZE;
//   params.wg_size = HOUSE_WORKGROUP_SIZE;
//   params.wg_count = HOUSE_WORKGROUP_NBR;

//   data_t dataset[HOUSE_SIZE] = {{0}};
//   create_dataset(&params, dataset);

//   parallelized_regression(&params, dataset, &results->parallelized);
//   iterative_regression(&params, dataset, &results->iterative);
// }

static
void temperature_regression(results_t * results) {
  linear_param_t params;
  params.filename = TEMP_FILENAME;
  params.size = TEMP_SIZE;
  params.wg_size = TEMP_WORKGROUP_SIZE;
  params.wg_count = TEMP_WORKGROUP_NBR;

  data_t dataset[TEMP_SIZE] = {{0}};
  create_dataset(&params, dataset);

  parallelized_regression(&params, dataset, &results->parallelized);
  iterative_regression(&params, dataset, &results->iterative);
}

static
void print_results(results_t * results) {
  PRINT_RESULT("Parallelized", results->parallelized);
  PRINT_RESULT("Iterative", results->iterative);
}

static 
void write_results(results_t * results, const char * restricts) {
  FILE* file = fopen(RESULT_FILENAME, restricts);
  WRITE_RESULT(file, results->parallelized);
  WRITE_RESULT(file, results->iterative);
  fclose(file);
}

int main(int argc, char* argv[]) {
    results_t results = {{0}};
    if (argc != 3) {
        fprintf(stderr, "usage: linear <num of loops> <cpu offset>\b");
        exit(0);
    }

    int loops = atoi(argv[1]);
    cpu_offset = atoi(argv[2]);

    init_opencl();
    double starttime = gettime();
    for (int i = 0; i < loops; i++) {
        temperature_regression(&results);
        //data2_regression(&results);
        write_results(&results, "a");
    }
    double endtime = gettime();
    printf("CPU offset: %d\n", cpu_offset);
    printf("Time: %lf ms\n", 1000.0 * (endtime - starttime));

  write_results(&results, "a");

  if (argc == 1 || strcmp(argv[1], "-no_print") > 0) {
    printf("\n> TEMPERATURE REGRESSION (%d)\n\n", TEMP_SIZE);
    print_results(&results);
  }

  free_opencl();

  return 0;
}
