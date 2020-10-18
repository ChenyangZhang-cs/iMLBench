#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <string>
#include <limits.h>
#include <fcntl.h>
#include <omp.h>
#include <unistd.h>
#include "kmeans.h"

#define _CRT_SECURE_NO_DEPRECATE 1


#ifdef WIN
	#include <windows.h>
#else
	#include <pthread.h>
	#include <sys/time.h>
	double gettime() {
		struct timeval t;
		gettimeofday(&t,NULL);
		return t.tv_sec+t.tv_usec*1e-6;
	}
#endif


#ifdef NV 
	#include <oclUtils.h>
#else
	#include <CL/cl.h>
#endif

#ifndef FLT_MAX
#define FLT_MAX 3.40282347e+38
#endif

#ifdef RD_WG_SIZE_0_0
        #define BLOCK_SIZE RD_WG_SIZE_0_0
#elif defined(RD_WG_SIZE_0)
        #define BLOCK_SIZE RD_WG_SIZE_0
#elif defined(RD_WG_SIZE)
        #define BLOCK_SIZE RD_WG_SIZE
#else
        #define BLOCK_SIZE 256
#endif

#ifdef RD_WG_SIZE_1_0
     #define BLOCK_SIZE2 RD_WG_SIZE_1_0
#elif defined(RD_WG_SIZE_1)
     #define BLOCK_SIZE2 RD_WG_SIZE_1
#elif defined(RD_WG_SIZE)
     #define BLOCK_SIZE2 RD_WG_SIZE
#else
     #define BLOCK_SIZE2 256
#endif

static int initialize(int use_gpu);
static int shutdown();
int allocate(int n_points, int n_features, int n_clusters, float **feature);
void deallocateMemory();
int	kmeansOCL(float **feature,    /* in: [npoints][nfeatures] */
           int     n_features,
           int     n_points,
           int     n_clusters,
           int    *membership,
		   float **clusters,
		   int     *new_centers_len,
           float  **new_centers);
// local variables
static cl_context	    context;
static cl_command_queue cmd_queue;
static cl_device_type   device_type;
static cl_device_id   * device_list;
static cl_int           num_devices;

// co_run
bool gpu_run = true, cpu_run = false;
int work_dim = 1;
size_t cpu_global_size[3] = {0, 1, 1};
size_t gpu_global_size[3] = {0, 1, 1};
size_t global_offset[2] = {0, 0};
size_t global_work[3] = { 0, 1 , 1 };
size_t local_work_size= BLOCK_SIZE; 

int cpu_offset = 0;

static int initialize(int use_gpu)
{
	cl_int result;
	size_t size;

	// create OpenCL context
	cl_platform_id platform_id;
	if (clGetPlatformIDs(1, &platform_id, NULL) != CL_SUCCESS) { printf("ERROR: clGetPlatformIDs(1,*,0) failed\n"); return -1; }
	cl_context_properties ctxprop[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platform_id, 0};
	device_type = use_gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU;
	context = clCreateContextFromType( ctxprop, device_type, NULL, NULL, NULL );
	if( !context ) { printf("ERROR: clCreateContextFromType(%s) failed\n", use_gpu ? "GPU" : "CPU"); return -1; }

	// get the list of GPUs
	result = clGetContextInfo( context, CL_CONTEXT_DEVICES, 0, NULL, &size );
	num_devices = (int) (size / sizeof(cl_device_id));
	
	if( result != CL_SUCCESS || num_devices < 1 ) { printf("ERROR: clGetContextInfo() failed\n"); return -1; }
	device_list = new cl_device_id[num_devices];
	if( !device_list ) { printf("ERROR: new cl_device_id[] failed\n"); return -1; }
	result = clGetContextInfo( context, CL_CONTEXT_DEVICES, size, device_list, NULL );
	if( result != CL_SUCCESS ) { printf("ERROR: clGetContextInfo() failed\n"); return -1; }

	// create command queue for the first device
	cmd_queue = clCreateCommandQueue( context, device_list[0], 0, NULL );
	if( !cmd_queue ) { printf("ERROR: clCreateCommandQueue() failed\n"); return -1; }



/*  check device CL support 
	cl_device_svm_capabilities caps;
for(int i = 0; i < num_devices; i++){
	cl_device_svm_capabilities caps;
cl_int err = clGetDeviceInfo(
    device_list[i],
    CL_DEVICE_SVM_CAPABILITIES,
    sizeof(cl_device_svm_capabilities),
    &caps,
    0
  );
if(err != CL_SUCCESS){
	printf("ERROR1\n");
}
else{
	if(caps & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER)
		printf("TYPE 1\n");
	else if(caps & CL_DEVICE_SVM_FINE_GRAIN_BUFFER)
		printf("TYPE 2\n");
	else if((caps & CL_DEVICE_SVM_FINE_GRAIN_BUFFER) && (caps & CL_DEVICE_SVM_ATOMICS))
		printf("TYPE 3\n");
	else if(caps & CL_DEVICE_SVM_FINE_GRAIN_SYSTEM)
		printf("TYPE 4\n");
	else if((caps & CL_DEVICE_SVM_FINE_GRAIN_SYSTEM) && (caps & CL_DEVICE_SVM_ATOMICS))
		printf("TYPE 5\n");
}
}
*/
	return 0;
}

static int shutdown()
{
	// release resources
	if( cmd_queue ) clReleaseCommandQueue( cmd_queue );
	if( context ) clReleaseContext( context );
	if( device_list ) delete device_list;

	// reset all variables
	cmd_queue = 0;
	context = 0;
	device_list = 0;
	num_devices = 0;
	device_type = 0;

	return 0;
}

typedef void* Pointer;
Pointer d_feature;
Pointer d_feature_swap;
Pointer d_cluster;
Pointer d_membership;

cl_kernel kernel;
cl_kernel kernel_s;
cl_kernel kernel2;

int threadsNum = 8;


int   *membership_OCL;
int   *membership_d;
float *feature_d;
float *clusters_d;
float *center_d;

int allocate(int n_points, int n_features, int n_clusters, float **feature)
{

	int sourcesize = 1024*1024;
	char * source = (char *)calloc(sourcesize, sizeof(char)); 
	if(!source) { printf("ERROR: calloc(%d) failed\n", sourcesize); return -1; }

	// read the kernel core source
	char * tempchar = "./kmeans.cl";
	FILE * fp = fopen(tempchar, "rb"); 
	if(!fp) { printf("ERROR: unable to open '%s'\n", tempchar); return -1; }
	fread(source + strlen(source), sourcesize, 1, fp);
	fclose(fp);
		
	// OpenCL initialization

	// compile kernel
	cl_int err = 0;
	const char * slist[2] = { source, 0 };
	cl_program prog = clCreateProgramWithSource(context, 1, slist, NULL, &err);
	if(err != CL_SUCCESS) { printf("ERROR: clCreateProgramWithSource() => %d\n", err); return -1; }
	err = clBuildProgram(prog, 0, NULL, NULL, NULL, NULL);
	{ // show warnings/errors
	//	static char log[65536]; memset(log, 0, sizeof(log));
	//	cl_device_id device_id = 0;
	//	err = clGetContextInfo(context, CL_CONTEXT_DEVICES, sizeof(device_id), &device_id, NULL);
	//	clGetProgramBuildInfo(prog, device_id, CL_PROGRAM_BUILD_LOG, sizeof(log)-1, log, NULL);
	//	if(err || strstr(log,"warning:") || strstr(log, "error:")) printf("<<<<\n%s\n>>>>\n", log);
	}
	if(err != CL_SUCCESS) { printf("ERROR: clBuildProgram() => %d\n", err); return -1; }
	
	char * kernel_kmeans_c  = "kmeans_kernel_c";
	char * kernel_swap  = "kmeans_swap";	
		
	kernel_s = clCreateKernel(prog, kernel_kmeans_c, &err);  
	if(err != CL_SUCCESS) { printf("ERROR: clCreateKernel() 0 => %d\n", err); return -1; }
	kernel2 = clCreateKernel(prog, kernel_swap, &err);  
	if(err != CL_SUCCESS) { printf("ERROR: clCreateKernel() 0 => %d\n", err); return -1; }
		
	clReleaseProgram(prog);	
	
//	d_feature = clSVMAlloc(context, CL_MEM_READ_WRITE, n_points * n_features * sizeof(float), 0 );
//	if(d_feature == NULL) { printf("ERROR: clSVMAlloc d_feature (size:%d)\n", n_points * n_features); return -1;}
	d_feature_swap = clSVMAlloc(context, CL_MEM_READ_WRITE, n_points * n_features * sizeof(float), 0);
	if(d_feature_swap == NULL) { printf("ERROR: clSVMAlloc d_feature_swap (size:%d) \n", n_points * n_features); return -1;}
	d_cluster = clSVMAlloc(context, CL_MEM_READ_WRITE, n_clusters * n_features  * sizeof(float), 0);
	if(d_cluster == NULL) { printf("ERROR: clSVMAlloc d_cluster (size:%d) \n", n_clusters * n_features); return -1;}
	d_membership = clSVMAlloc(context, CL_MEM_READ_WRITE, n_points * sizeof(int), 0 );
	if(d_membership == NULL) { printf("ERROR: clSVMAlloc d_membership (size:%d) \n", n_points); return -1;}
			
	//write buffers
//	err = clEnqueueSVMMap(cmd_queue, CL_TRUE, CL_MAP_WRITE, d_feature, n_points * n_features * sizeof(float), 0, 0, 0);
	//err &= clEnqueueWriteBuffer(cmd_queue, (cl_mem)d_feature, 1, 0, n_points * n_features * sizeof(float), feature[0], 0, 0, 0);
//	memmove(d_feature, feature[0], n_points * n_features * sizeof(float));
//	err &= clEnqueueSVMUnmap(cmd_queue, d_feature, 0, 0, 0);

	if(err != CL_SUCCESS) { printf("ERROR: clEnqueueWriteBuffer d_feature (size:%d) => %d\n", n_points * n_features, err); return -1; }

/*
	err = clSetKernelArgSVMPointer(kernel2, 0, (void*) d_feature);
	err &= clSetKernelArgSVMPointer(kernel2, 1, (void*) d_feature_swap);
	err &= clSetKernelArgSVMPointer(kernel2, 2, (void*) d_n_points);
	err &= clSetKernelArgSVMPointer(kernel2, 3, (void*) d_n_features);
*/

	err = clSetKernelArgSVMPointer(kernel2, 0, (void*) d_feature);
	err &= clSetKernelArgSVMPointer(kernel2, 1, (void*) d_feature_swap);
	err &= clSetKernelArg(kernel2, 2, sizeof(cl_int), (void*) &n_points);
	err &= clSetKernelArg(kernel2, 3, sizeof(cl_int), (void*) &n_features);

	err = clEnqueueNDRangeKernel(cmd_queue, kernel2, 1, NULL, global_work, &local_work_size, 0, 0, 0);
	// err = clEnqueueNDRangeKernel(cmd_queue, kernel2, 1, NULL, global_work, &local_work_size, 0, 0, 0);

	if(err != CL_SUCCESS) { printf("ERROR1: clEnqueueNDRangeKernel()=>%d, globalsize: %d,localSize: %d,maxsize:%d failed\n", 
		err, global_work[0] * global_work[1], local_work_size, CL_DEVICE_MAX_WORK_ITEM_SIZES ); return -1; }
	
//	membership_OCL = (int*) malloc(n_points * sizeof(int));
}

void deallocateMemory()
{
	/*
	clReleaseMemObject(d_feature);
	clReleaseMemObject(d_feature_swap);
	clReleaseMemObject(d_cluster);
	clReleaseMemObject(d_membership);
	*/
	clSVMFree(context, d_feature);
	clSVMFree(context, d_feature_swap);
	clSVMFree(context, d_cluster);
	clSVMFree(context, d_membership);

//	free(membership_OCL);
}


int main( int argc, char** argv) 
{
	printf("WG size of kernel_swap = %d, WG size of kernel_kmeans = %d, work size = %d \n", BLOCK_SIZE, BLOCK_SIZE2,CL_DEVICE_MAX_WORK_GROUP_SIZE);
	int corenum = omp_get_num_procs();
	printf("Core number: %d\n", corenum);
	setup(argc, argv);
	shutdown();
}

//	((float *) d_feature_swap, (float *) d_cluster, (int *) d_membership, n_points, cpu_global_size[0], n_clusters, n_features, 0, 0); 

void kmeansOMP(float  *feature,   
			   float  *clusters,
			    int    *membership,
			    int     npoints,
			    int     cpusize,
				int     nclusters,
				int     nfeatures,
				int		offset,
				int		size
			  ) 
{

	//printf("OMP size: %d\n",cpusize);
#pragma omp parallel for num_threads(threadsNum)
	for (int j = 0; j < cpusize; j++) {
		/*
		int num = omp_get_num_threads();
		int rank = omp_get_thread_num();
		printf("Total num: %d, Rank : %d , j = %d\n",num, rank, j);
*/
		int index = 0;
		float min_dist=FLT_MAX;
		//printf("\npoint: %d\n", j);
//#pragma omp parallel for shared(min_dist, index)
		for(int i = 0; i < nclusters; i++){
			float dist = 0;
			float ans  = 0;
#pragma omp simd reduction(+:ans) 
			for (int l = 0; l < nfeatures; l++){
				float tmp = feature[l * npoints + j] - clusters[i * nfeatures + l];
				ans += tmp * tmp;			
			}
			dist = ans;   
			if (dist < min_dist) {
				min_dist = dist;
				index    = i;	
			}
		}
		membership[j] = index;
	}	
	return;
}

int	kmeansOCL(float **feature,    /* in: [npoints][nfeatures] */
           int     n_features,
           int     n_points,
           int     n_clusters,
           int    *membership,
		   float **clusters,
		   int     *new_centers_len,
           float  **new_centers)	
{
  
	int delta = 0;
	int i, j, k;
	cl_int err = 0;
	
//	size_t global_work[3] = { n_points, 1, 1 }; 

	/// Ke Wang adjustable local group size 2013/08/07 10:37:33
/*
	size_t local_work_size=BLOCK_SIZE2; // work group size is defined by RD_WG_SIZE_1 or RD_WG_SIZE_1_0 2014/06/10 17:00:41
	if(global_work[0]%local_work_size !=0)
	  global_work[0]=(global_work[0]/local_work_size+1)*local_work_size;
*/
//	err = clEnqueueSVMMap(cmd_queue, CL_TRUE, CL_MAP_WRITE, d_cluster, n_clusters * n_features * sizeof(float), 0, 0, 0);
	memmove(d_cluster, clusters[0], n_clusters * n_features * sizeof(float));
//	if(err != CL_SUCCESS) { printf("ERROR: clEnqueueWriteBuffer d_cluster (size:%d) => %d\n", n_points, err); return -1; }
//	err &= clEnqueueSVMUnmap(cmd_queue, d_membership, 0, 0, 0);

/*
 *   co-run
 */
    cl_event eventList;
//double t1 = gettime();
	if(gpu_run){
		int size = 0; int offset = 0;

		clSetKernelArgSVMPointer(kernel_s, 0, (void*) d_feature_swap);
		clSetKernelArgSVMPointer(kernel_s, 1, (void*) d_cluster);
		clSetKernelArgSVMPointer(kernel_s, 2, (void*) d_membership);
		clSetKernelArg(kernel_s, 3, sizeof(cl_int), (void*) &n_points);
		clSetKernelArg(kernel_s, 4, sizeof(cl_int), (void*) &n_clusters);
		clSetKernelArg(kernel_s, 5, sizeof(cl_int), (void*) &n_features);
		clSetKernelArg(kernel_s, 6, sizeof(cl_int), (void*) &offset);
		clSetKernelArg(kernel_s, 7, sizeof(cl_int), (void*) &size);
		
		err = clEnqueueNDRangeKernel(cmd_queue, kernel_s, work_dim, global_offset, gpu_global_size, &local_work_size, 0, 0, &eventList);
		if(err != CL_SUCCESS) printf("ERROR 1 in corun\n");
	}

	if(cpu_run){
// cluster and features are read, membership is written
		/*
		err = clEnqueueSVMMap(cmd_queue, CL_TRUE, CL_MAP_WRITE, d_membership, n_points * sizeof(int), 0, 0, 0);
		err &= clEnqueueSVMMap(cmd_queue, CL_TRUE, CL_MAP_READ, d_cluster, n_clusters * n_features * sizeof(float), 0, 0, 0);
		err &= clEnqueueSVMMap(cmd_queue, CL_TRUE, CL_MAP_READ, d_feature_swap, n_points * n_features * sizeof(float), 0, 0, 0);
		*/
	//	double time1 = gettime();
		kmeansOMP((float *) d_feature_swap, (float *) d_cluster, (int *) d_membership, n_points, cpu_global_size[0], n_clusters, n_features, 0, 0); 
	//	double time2 = gettime();
	//	printf("OMP time: %lf ms\n", 1000.0*(time2 - time1));
		/*
		err &= clEnqueueSVMUnmap(cmd_queue, d_membership, 0, 0, 0);
		err &= clEnqueueSVMUnmap(cmd_queue, d_cluster, 0, 0, 0);
		err &= clEnqueueSVMUnmap(cmd_queue, d_feature_swap, 0, 0, 0);
		*/
	}

	if(gpu_run){
		err = clWaitForEvents(1,&eventList);
		if(err != CL_SUCCESS)	printf("ERROR 2 in corun\n");
	} 

/*
	for(i = 0; i < n_points; i++)
		printf("%d, ", membership_OCL[i]);
	printf("\n\n");
*/
//double t2 = gettime();
//printf("One iteration time: %f ms\n", 1000*(t2 - t1));
	membership_OCL = (int *)d_membership;   
//	err = clEnqueueSVMMap(cmd_queue, CL_TRUE, CL_MAP_READ, d_membership, n_points * sizeof(int), 0, 0, 0);

	delta = 0;
	for (i = 0; i < n_points; i++)
	{
		int cluster_id = membership_OCL[i];
		new_centers_len[cluster_id]++;
		if (membership_OCL[i] != membership[i])
		{
			delta++;
			membership[i] = membership_OCL[i];
		}
		for (j = 0; j < n_features; j++)
		{
			new_centers[cluster_id][j] += feature[i][j];
		}
	}
//	err &= clEnqueueSVMUnmap(cmd_queue, d_membership, 0, 0, 0);


	if(err != CL_SUCCESS) { printf("ERROR: Memcopy Out\n"); return -1; }
	clFinish(cmd_queue);

	return delta;
}


//extern double wtime(void);

/*---< usage() >------------------------------------------------------------*/
void usage(char *argv0) {
    char *help =
        "\nUsage: %s [switches] -i filename\n\n"
		"    -i filename      :file containing data to be clustered\n"		
		"    -m max_nclusters :maximum number of clusters allowed    [default=5]\n"
        "    -n min_nclusters :minimum number of clusters allowed    [default=5]\n"
		"    -t threshold     :threshold value                       [default=0.001]\n"
		"    -l nloops        :iteration for each number of clusters [default=1]\n"
		"    -b               :input file is in binary format\n"
        "    -r               :calculate RMSE                        [default=off]\n"
		"    -o               :output cluster center coordinates     [default=off]\n";
    fprintf(stderr, help, argv0);
    exit(-1);
}

/*---< main() >-------------------------------------------------------------*/
int setup(int argc, char **argv) {
		int		opt;
 extern char   *optarg;
		char   *filename = 0;
		float  *buf;
		char	line[1024];
		int		isBinaryFile = 0;

		float	threshold = 0.001;		/* default value */
		int		max_nclusters=5;		/* default value */
		int		min_nclusters=5;		/* default value */
		int		best_nclusters = 0;
		int		nfeatures = 0;
		int		npoints = 0;
		float	len;
		         
		float **features;
		float **cluster_centres=NULL;
		int		i, j, index;
		int		nloops = 1;				/* default value */
				
		int		isRMSE = 0;		
		float	rmse;
		
		int		isOutput = 0;
		//float	cluster_timing, io_timing;		

		/* obtain command line arguments and change appropriate options */
		while ( (opt=getopt(argc,argv,"s:f:i:t:m:n:l:bro"))!= EOF) {
        switch (opt) {
            case 'i': filename=optarg;
                      break;
            case 'b': isBinaryFile = 1;
                      break;            
            case 't': threshold=atof(optarg);
                      break;
            case 'm': max_nclusters = atoi(optarg);
                      break;
            case 'n': min_nclusters = atoi(optarg);
                      break;
			case 'r': isRMSE = 1;
                      break;
			case 'o': isOutput = 1;
					  break;
		    case 'l': nloops = atoi(optarg);
					  break;
			case 's': threadsNum = atoi(optarg);
					  break;
			case 'f': cpu_offset = atoi(optarg);
                      printf("cpu_offset = %d\n", cpu_offset);
                      break;
            case '?': usage(argv[0]);
                      break;
            default: usage(argv[0]);
                      break;
        }
    }

    if (filename == 0) usage(argv[0]);

    int use_gpu = 1;
	initialize(use_gpu);
//	if(context == -1) return -1;

		
	/* ============== I/O begin ==============*/
    /* get nfeatures and npoints */
    //io_timing = omp_get_wtime();
    if (isBinaryFile) {		//Binary file input
        int infile;
        if ((infile = open(filename, O_RDONLY, "0600")) == -1) {
            fprintf(stderr, "Error: no such file (%s)\n", filename);
            exit(1);
        }
        read(infile, &npoints,   sizeof(int));
        read(infile, &nfeatures, sizeof(int));        

        /* allocate space for features[][] and read attributes of all objects */
        buf         = (float*) malloc(npoints*nfeatures*sizeof(float));
        features    = (float**)malloc(npoints*          sizeof(float*));
        features[0] = (float*)clSVMAlloc(context, CL_MEM_READ_WRITE, npoints * nfeatures * sizeof(float), 0 );
        //features[0] = (float*) malloc(npoints*nfeatures*sizeof(float));
        for (i=1; i<npoints; i++)
            features[i] = features[i-1] + nfeatures;

        read(infile, buf, npoints*nfeatures*sizeof(float));

        close(infile);
    }
    else {
        FILE *infile;
        if ((infile = fopen(filename, "r")) == NULL) {
            fprintf(stderr, "Error: no such file (%s)\n", filename);
            exit(1);
		}		
        while (fgets(line, 1024, infile) != NULL)
			if (strtok(line, " \t\n") != 0)
                npoints++;			
        rewind(infile);
        while (fgets(line, 1024, infile) != NULL) {
            if (strtok(line, " \t\n") != 0) {
                /* ignore the id (first attribute): nfeatures = 1; */
                while (strtok(NULL, " ,\t\n") != NULL) nfeatures++;
                break;
            }
        }        

        /* allocate space for features[] and read attributes of all objects */
        buf         = (float*) malloc(npoints*nfeatures*sizeof(float));
        features    = (float**)malloc(npoints*          sizeof(float*));
        features[0] = (float*)clSVMAlloc(context, CL_MEM_READ_WRITE, npoints * nfeatures * sizeof(float), 0 );
//        features[0] = (float*) malloc(npoints*nfeatures*sizeof(float));
        for (i=1; i<npoints; i++)
            features[i] = features[i-1] + nfeatures;
        rewind(infile);
        i = 0;
        while (fgets(line, 1024, infile) != NULL) {
            if (strtok(line, " \t\n") == NULL) continue;            
            for (j=0; j<nfeatures; j++) {
                buf[i] = atof(strtok(NULL, " ,\t\n"));             
                i++;
            }            
        }
        fclose(infile);
    }
    //io_timing = omp_get_wtime() - io_timing;
	
	printf("\nI/O completed\n");
	printf("\nNumber of objects: %d\n", npoints);
	printf("Number of features: %d\n", nfeatures);	
	/* ============== I/O end ==============*/

		// create co_run
	global_work[0] = npoints;
	if(global_work[0]%local_work_size !=0)
	    global_work[0]=(global_work[0]/local_work_size+1)*local_work_size;

	if(cpu_offset > 0){
		cpu_run = true;
	}
	cpu_global_size[0] = cpu_offset * global_work[0] / 100;
	if(cpu_global_size[0] % local_work_size != 0){
		cpu_global_size[0] = (1 + cpu_global_size[0] / local_work_size) * local_work_size;
	}
	gpu_global_size[0] = global_work[0] - cpu_global_size[0];
	if(gpu_global_size[0] <= 0){
		gpu_run = false;
	}
	global_offset[0] = cpu_global_size[0];
	printf("CPU size: %d, GPU size: %d\n", cpu_global_size[0], gpu_global_size[0]);


	// error check for clusters
	if (npoints < min_nclusters)
	{
		printf("Error: min_nclusters(%d) > npoints(%d) -- cannot proceed\n", min_nclusters, npoints);
		exit(0);
	}

	srand(7);												/* seed for future random number generator */
//	clEnqueueSVMMap(cmd_queue, CL_TRUE, CL_MAP_WRITE, features[0], npoints * nfeatures * sizeof(float), 0, 0, 0);
	//err &= clEnqueueWriteBuffer(cmd_queue, (cl_mem)d_feature, 1, 0, n_points * n_features * sizeof(float), feature[0], 0, 0, 0);
//	memmove(d_feature, feature[0], n_points * n_features * sizeof(float));
	memcpy(features[0], buf, npoints*nfeatures*sizeof(float)); /* now features holds 2-dimensional array of features */
//	clEnqueueSVMUnmap(cmd_queue, features[0], 0, 0, 0);

	d_feature = features[0];
	free(buf);

	/* ======================= core of the clustering ===================*/

    //cluster_timing = omp_get_wtime();		/* Total clustering time */
	cluster_centres = NULL;
	double startTime = gettime();
    index = cluster(npoints,				/* number of data points */
					nfeatures,				/* number of features for each point */
					features,				/* array: [npoints][nfeatures] */
					min_nclusters,			/* range of min to max number of clusters */
					max_nclusters,
					threshold,				/* loop termination factor */
				   &best_nclusters,			/* return: number between min and max */
				   &cluster_centres,		/* return: [best_nclusters][nfeatures] */  
				   &rmse,					/* Root Mean Squared Error */
					isRMSE,					/* calculate RMSE */
					nloops);				/* number of iteration for each number of clusters */		
    double endTime = gettime();
    printf("Time: %lf\n", 1000.0*(endTime - startTime));

	//cluster_timing = omp_get_wtime() - cluster_timing;


	/* =============== Command Line Output =============== */

	/* cluster center coordinates
	   :displayed only for when k=1*/
	if((min_nclusters == max_nclusters) && (isOutput == 1)) {
		printf("\n================= Centroid Coordinates =================\n");
		for(i = 0; i < max_nclusters; i++){
			printf("%d:", i);
			for(j = 0; j < nfeatures; j++){
				printf(" %.2f", cluster_centres[i][j]);
			}
			printf("\n\n");
		}
	}
	
	len = (float) ((max_nclusters - min_nclusters + 1)*nloops);

	printf("Number of Iteration: %d\n", nloops);
	//printf("Time for I/O: %.5fsec\n", io_timing);
	//printf("Time for Entire Clustering: %.5fsec\n", cluster_timing);
	
	if(min_nclusters != max_nclusters){
		if(nloops != 1){									//range of k, multiple iteration
			//printf("Average Clustering Time: %fsec\n",
			//		cluster_timing / len);
			printf("Best number of clusters is %d\n", best_nclusters);				
		}
		else{												//range of k, single iteration
			//printf("Average Clustering Time: %fsec\n",
			//		cluster_timing / len);
			printf("Best number of clusters is %d\n", best_nclusters);				
		}
	}
	else{
		if(nloops != 1){									// single k, multiple iteration
			//printf("Average Clustering Time: %.5fsec\n",
			//		cluster_timing / nloops);
			if(isRMSE)										// if calculated RMSE
				printf("Number of trials to approach the best RMSE of %.3f is %d\n", rmse, index + 1);
		}
		else{												// single k, single iteration				
			if(isRMSE)										// if calculated RMSE
				printf("Root Mean Squared Error: %.3f\n", rmse);
		}
	}
	

	/* free up memory */
//	free(features[0]);

	free(features);    
    return(0);
}
