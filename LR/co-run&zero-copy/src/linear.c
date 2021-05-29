#include "linear.h"
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
clock_t start;
clock_t end;
int loops = 10;

double gettime() {
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec + t.tv_usec * 1e-6;
}

extern void create_dataset(linear_param_t* params, data_t* dataset);
extern void temperature_regression(results_t* results);

extern int cpu_offset;

static void print_results(results_t* results) {
    PRINT_RESULT("Parallelized", results->parallelized);
    PRINT_RESULT("Iterative", results->iterative);
}

static void write_results(results_t* results, const char* restricts) {
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

    loops = atoi(argv[1]);
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
    if (argc == 1 || strcmp(argv[1], "-no_print") > 0) {
        printf("> TEMPERATURE REGRESSION (%d)\n\n", TEMP_SIZE);
        print_results(&results);
    }

    free_opencl();

    return 0;
}
