#ifndef __NEAREST_NEIGHBOR__
#define __NEAREST_NEIGHBOR__

#include "nearestNeighbor.h"
#include <math.h>
#include <sys/time.h>
#include <omp.h>

cl_context context = NULL;
int cpu_offset = 0;
int loops = 1;

double gettime() {
  struct timeval t;
  gettimeofday(&t,NULL);
  return t.tv_sec+t.tv_usec*1e-6;
}

void NearestNeighborOMP(LatLong *d_locations,
                        float *d_distances,
                        const int numRecords,
                        const float lat,
                        const float lng)
{
//    double time1 = gettime();
//#pragma omp parallel for
    for (int i = 0; i < numRecords; i++)
    {
        LatLong *latLong = d_locations + i;
        float *dist = d_distances + i;
        *dist = (float)sqrt((lat - latLong->lat) * (lat - latLong->lat) + (lng - latLong->lng) * (lng - latLong->lng));
    }
}


int main(int argc, char *argv[])
{
    std::vector<Record> records;
    float *recordDistances;
    //LatLong locations[REC_WINDOW];
    std::vector<LatLong> locations;
    int i;
    // args
    char filename[100];
    int resultsCount = 10, quiet = 0, timing = 0, platform = -1, device = -1;
    float lat = 0.0, lng = 0.0;

    // parse command line
    if (parseCommandline(argc, argv, filename, &resultsCount, &lat, &lng,
                         &quiet, &timing, &platform, &device))
    {
        printUsage();
        return 0;
    }

    int numRecords = loadData(filename, records, locations);

    //for(i=0;i<numRecords;i++)
    //    printf("%s, %f, %f\n",(records[i].recString),locations[i].lat,locations[i].lng);

    if (!quiet)
    {
        printf("Number of records: %d\n", numRecords);
        printf("Finding the %d closest neighbors.\n", resultsCount);
    }

    if (resultsCount > numRecords)
        resultsCount = numRecords;

    context = cl_init_context(platform, device, quiet);

    double starttime = gettime();
    for(int i = 0; i < 5; i++){
        recordDistances = OpenClFindNearestNeighbors(context, numRecords, locations, lat, lng, timing);
        // find the resultsCount least distances
        findLowest(records, recordDistances, numRecords, resultsCount);
    }
    double endtime = gettime();
    printf("Time: %lf\n", 1000.0*(endtime - starttime));
    // print out results
    if (!quiet)
        for (i = 0; i < resultsCount; i++)
        {
            printf("%s --> Distance=%f\n", records[i].recString, records[i].distance);
        }
    clSVMFree(context, recordDistances);
    return 0;
}

float *OpenClFindNearestNeighbors(
    cl_context context,
    int numRecords,
    std::vector<LatLong> &locations, float lat, float lng,
    int timing)
{

    // 1. set up kernel
    cl_kernel NN_kernel;
    cl_int status;
    cl_program cl_NN_program;
    cl_NN_program = cl_compileProgram(
        (char *)"nearestNeighbor_kernel.cl", NULL);

    NN_kernel = clCreateKernel(
        cl_NN_program, "NearestNeighbor", &status);
    status = cl_errChk(status, (char *)"Error Creating Nearest Neighbor kernel", true);
    if (status)
        exit(1);

    // 2. set up memory on device and send ipts data to device
    // copy ipts(1,2) to device
    // also need to alloate memory for the distancePoints
    /*
    cl_mem d_locations;
    cl_mem d_distances;
    d_locations = clCreateBuffer(context, CL_MEM_READ_ONLY,
        sizeof(LatLong) * numRecords, NULL, &error);

    d_distances = clCreateBuffer(context, CL_MEM_READ_WRITE,
        sizeof(float) * numRecords, NULL, &error);
        */

    cl_int error = 0;
    void *d_locations = clSVMAlloc(context, CL_MEM_READ_ONLY, sizeof(LatLong) * numRecords, 0);
    void *d_distances = clSVMAlloc(context, CL_MEM_READ_WRITE, sizeof(float) * numRecords, 0);

    cl_command_queue command_queue = cl_getCommandQueue();
    cl_event writeEvent, kernelEvent, readEvent;
    /*
    error = clEnqueueWriteBuffer(command_queue,
               d_locations,
               1, // change to 0 for nonblocking write
               0, // offset
               sizeof(LatLong) * numRecords,
               &locations[0],
               0,
               NULL,
               &writeEvent);*/
    memcpy(d_locations, &locations[0], sizeof(LatLong) * numRecords);

    // 3. send arguments to device

    // 4. enqueue kernel
    size_t globalWorkSize[1];
    globalWorkSize[0] = numRecords;
    if (numRecords % 64)
        globalWorkSize[0] += 64 - (numRecords % 64);
    //printf("Global Work Size: %zu\n",globalWorkSize[0]);
    // Co_run
    bool gpu_run = true, cpu_run = false;
    int work_dim = 1;
    size_t cpu_global_size[3] = {0, 1, 1};
    size_t gpu_global_size[3] = {0, 1, 1};
    size_t global_offset[2] = {0, 0};
    size_t global_work[3] = {0, 1, 1};

    if (cpu_offset > 0){
        cpu_run = true;
    }
    cpu_global_size[0] = cpu_offset * globalWorkSize[0] / 100;
    if (cpu_global_size[0] % 64 != 0)
    {
        cpu_global_size[0] = (1 + cpu_global_size[0] / 64) * 64;
    }
    gpu_global_size[0] = globalWorkSize[0] - cpu_global_size[0];
    if (gpu_global_size[0] <= 0)
    {
        gpu_run = false;
    }
    global_offset[0] = cpu_global_size[0];
//    printf("CPU size: %d, GPU size: %d\n", cpu_global_size[0], gpu_global_size[0]);

    float * distance_cpu = (float *)malloc(sizeof(float) * cpu_global_size[0]);

    double tstart = gettime();
    for(int i = 0; i < 10000; i++){


    if (gpu_run)
    {
        cl_int argchk;
        double t1 = gettime();
        argchk = clSetKernelArgSVMPointer(NN_kernel, 0, (void *)d_locations);
        argchk |= clSetKernelArgSVMPointer(NN_kernel, 1, (void *)d_distances);
        argchk |= clSetKernelArg(NN_kernel, 2, sizeof(int), (void *)&numRecords);
        argchk |= clSetKernelArg(NN_kernel, 3, sizeof(float), (void *)&lat);
        argchk |= clSetKernelArg(NN_kernel, 4, sizeof(float), (void *)&lng);
        cl_errChk(argchk, "ERROR in Setting Nearest Neighbor kernel args", true);
        error = clEnqueueNDRangeKernel(command_queue, NN_kernel, 1, global_offset, gpu_global_size, NULL, 0, NULL, &kernelEvent);
        double tstart = gettime();

        cl_errChk(error, "ERROR in Executing Kernel NearestNeighbor", true);

     //   clWaitForEvents(1,&kernelEvent);
        double t2 = gettime();

       // printf("GPU time: %lf ms\n", 1000*(t2-t1));

    }
    if (cpu_run)
    {
      //printf("numRecords:%d\n",numRecords);
      //printf("lat:%lf\n",lat);
      //printf("lng:%lf\n",lng);
              double t1 = gettime();

        NearestNeighborOMP((LatLong *)d_locations, distance_cpu, cpu_global_size[0], lat, lng);
        double t2 = gettime();
        //printf("GPU time: %lf ms\n", 1000*(t2-t1));
    }
    
    if(gpu_run)
    {
		cl_int err = clWaitForEvents(1,&kernelEvent);
		if(err != CL_SUCCESS)	printf("ERROR in corun\n");
	} 
    
  }
    double tend = gettime();
    printf("Training Time: %lf\n", 1000.0*(tend - tstart));

    memcpy(d_distances, distance_cpu, sizeof(float) * cpu_global_size[0]);

    // 5. transfer data off of device

    // create distances std::vector
    float *distances = (float *)d_distances;
    /*
    error = clEnqueueReadBuffer(command_queue,
        d_distances,
        1, // change to 0 for nonblocking write
        0, // offset
        sizeof(float) * numRecords,
        distances,
        0,
        NULL,
        &readEvent);*/

    if (timing)
    {
        clFinish(command_queue);
        cl_ulong eventStart, eventEnd, totalTime = 0;
        printf("# Records\tWrite(s) [size]\t\tKernel(s)\tRead(s)  [size]\t\tTotal(s)\n");
        printf("%d        \t", numRecords);
        // Write Buffer
        error = clGetEventProfilingInfo(writeEvent, CL_PROFILING_COMMAND_START,
                                        sizeof(cl_ulong), &eventStart, NULL);
        cl_errChk(error, "ERROR in Event Profiling (Write Start)", true);
        error = clGetEventProfilingInfo(writeEvent, CL_PROFILING_COMMAND_END,
                                        sizeof(cl_ulong), &eventEnd, NULL);
        cl_errChk(error, "ERROR in Event Profiling (Write End)", true);

        printf("%f [%.2fMB]\t", (float)((eventEnd - eventStart) / 1e9), (float)((sizeof(LatLong) * numRecords) / 1e6));
        totalTime += eventEnd - eventStart;
        // Kernel
        error = clGetEventProfilingInfo(kernelEvent, CL_PROFILING_COMMAND_START,
                                        sizeof(cl_ulong), &eventStart, NULL);
        cl_errChk(error, "ERROR in Event Profiling (Kernel Start)", true);
        error = clGetEventProfilingInfo(kernelEvent, CL_PROFILING_COMMAND_END,
                                        sizeof(cl_ulong), &eventEnd, NULL);
        cl_errChk(error, "ERROR in Event Profiling (Kernel End)", true);

        printf("%f\t", (float)((eventEnd - eventStart) / 1e9));
        totalTime += eventEnd - eventStart;
        // Read Buffer
        error = clGetEventProfilingInfo(readEvent, CL_PROFILING_COMMAND_START,
                                        sizeof(cl_ulong), &eventStart, NULL);
        cl_errChk(error, "ERROR in Event Profiling (Read Start)", true);
        error = clGetEventProfilingInfo(readEvent, CL_PROFILING_COMMAND_END,
                                        sizeof(cl_ulong), &eventEnd, NULL);
        cl_errChk(error, "ERROR in Event Profiling (Read End)", true);

        printf("%f [%.2fMB]\t", (float)((eventEnd - eventStart) / 1e9), (float)((sizeof(float) * numRecords) / 1e6));
        totalTime += eventEnd - eventStart;

        printf("%f\n\n", (float)(totalTime / 1e9));
    }
    // 6. return finalized data and release buffers
    clSVMFree(context, d_locations);
    //clReleaseMemObject(d_distances);
    return distances;
}

int loadData(char *filename, std::vector<Record> &records, std::vector<LatLong> &locations)
{
    FILE *flist, *fp;
    int i = 0;
    char dbname[64];
    int recNum = 0;

    /**Main processing **/

    flist = fopen(filename, "r");
    while (!feof(flist))
    {
        /**
		* Read in REC_WINDOW records of length REC_LENGTH
		* If this is the last file in the filelist, then done
		* else open next file to be read next iteration
		*/
        if (fscanf(flist, "%s\n", dbname) != 1)
        {
            fprintf(stderr, "error reading filelist\n");
            exit(0);
        }
        fp = fopen(dbname, "r");
        if (!fp)
        {
            printf("error opening a db\n");
            exit(1);
        }
        // read each record
        while (!feof(fp))
        {
            Record record;
            LatLong latLong;
            fgets(record.recString, 49, fp);
            fgetc(fp); // newline
            if (feof(fp))
                break;

            // parse for lat and long
            char substr[6];

            for (i = 0; i < 5; i++)
                substr[i] = *(record.recString + i + 28);
            substr[5] = '\0';
            latLong.lat = atof(substr);

            for (i = 0; i < 5; i++)
                substr[i] = *(record.recString + i + 33);
            substr[5] = '\0';
            latLong.lng = atof(substr);

            locations.push_back(latLong);
            records.push_back(record);
            recNum++;
        }
        fclose(fp);
    }
    fclose(flist);
    return recNum;
}

void findLowest(std::vector<Record> &records, float *distances, int numRecords, int topN)
{
    int i, j;
    float val;
    int minLoc;
    Record *tempRec;
    float tempDist;

    for (i = 0; i < topN; i++)
    {
        minLoc = i;
        for (j = i; j < numRecords; j++)
        {
            val = distances[j];
            if (val < distances[minLoc])
                minLoc = j;
        }
        // swap locations and distances
        tempRec = &records[i];
        records[i] = records[minLoc];
        records[minLoc] = *tempRec;

        tempDist = distances[i];
        distances[i] = distances[minLoc];
        distances[minLoc] = tempDist;

        // add distance to the min we just found
        records[i].distance = distances[i];
    }
}

int parseCommandline(int argc, char *argv[], char *filename, int *r, float *lat, float *lng,
                     int *q, int *t, int *p, int *d)
{
    int i;
    if (argc < 2)
        return 1; // error
    strncpy(filename, argv[1], 100);
    char flag;

    for (i = 1; i < argc; i++)
    {
        if (argv[i][0] == '-')
        { // flag
            flag = argv[i][1];
            switch (flag)
            {
            case 'r': // number of results
                i++;
                *r = atoi(argv[i]);
                break;
            case 'l': // lat or lng
                if (argv[i][2] == 'a')
                { //lat
                    *lat = atof(argv[i + 1]);
                }
                else
                { //lng
                    *lng = atof(argv[i + 1]);
                }
                i++;
                break;
            case 'h': // help
                return 1;
                break;
            case 'q': // quiet
                *q = 1;
                break;
            case 't': // timing
                *t = 1;
                break;
            case 'p': // platform
                i++;
                *p = atoi(argv[i]);
                break;
            case 'd': // device
                i++;
                *d = atoi(argv[i]);
                break;
            case 'o': // cpu_offset
                i++;
                cpu_offset = atoi(argv[i]);
                break;
            case 'i':
                i++;
                loops = atoi(argv[i]);
                break;
            }
        }
    }
    if ((*d >= 0 && *p < 0) || (*p >= 0 && *d < 0)) // both p and d must be specified if either are specified
        return 1;
    return 0;
}

void printUsage()
{
    printf("Nearest Neighbor Usage\n");
    printf("\n");
    printf("nearestNeighbor [filename] -r [int] -lat [float] -lng [float] [-hqt] [-p [int] -d [int]]\n");
    printf("\n");
    printf("example:\n");
    printf("$ ./nearestNeighbor filelist.txt -r 5 -lat 30 -lng 90\n");
    printf("\n");
    printf("filename     the filename that lists the data input files\n");
    printf("-r [int]     the number of records to return (default: 10)\n");
    printf("-lat [float] the latitude for nearest neighbors (default: 0)\n");
    printf("-lng [float] the longitude for nearest neighbors (default: 0)\n");
    printf("\n");
    printf("-h, --help   Display the help file\n");
    printf("-q           Quiet mode. Suppress all text output.\n");
    printf("-t           Print timing information.\n");
    printf("\n");
    printf("-p [int]     Choose the platform (must choose both platform and device)\n");
    printf("-d [int]     Choose the device (must choose both platform and device)\n");
    printf("\n");
    printf("\n");
    printf("Notes: 1. The filename is required as the first parameter.\n");
    printf("       2. If you declare either the device or the platform,\n");
    printf("          you must declare both.\n\n");
}

#endif
