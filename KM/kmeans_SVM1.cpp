#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <string>
#include "kmeans.h"

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

/*
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
	int use_gpu = 1;
	if(initialize(use_gpu)) return -1;

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
	
	d_feature = clSVMAlloc(context, CL_MEM_READ_WRITE, n_points * n_features * sizeof(float), 0 );
	if(d_feature == NULL) { printf("ERROR: clSVMAlloc d_feature (size:%d)\n", n_points * n_features); return -1;}
	d_feature_swap = clSVMAlloc(context, CL_MEM_READ_WRITE, n_points * n_features * sizeof(float), 0);
	if(d_feature_swap == NULL) { printf("ERROR: clSVMAlloc d_feature_swap (size:%d) \n", n_points * n_features); return -1;}
	d_cluster = clSVMAlloc(context, CL_MEM_READ_WRITE, n_clusters * n_features  * sizeof(float), 0);
	if(d_cluster == NULL) { printf("ERROR: clSVMAlloc d_cluster (size:%d) \n", n_clusters * n_features); return -1;}
	d_membership = clSVMAlloc(context, CL_MEM_READ_WRITE, n_points * sizeof(int), 0 );
	if(d_membership == NULL) { printf("ERROR: clSVMAlloc d_membership (size:%d) \n", n_points); return -1;}
			
	//write buffers
	err = clEnqueueSVMMap(cmd_queue, CL_TRUE, CL_MAP_WRITE, d_feature, n_points * n_features * sizeof(float), 0, 0, 0);
	//err &= clEnqueueWriteBuffer(cmd_queue, (cl_mem)d_feature, 1, 0, n_points * n_features * sizeof(float), feature[0], 0, 0, 0);
	memmove(d_feature, feature[0], n_points * n_features * sizeof(float));
	err &= clEnqueueSVMUnmap(cmd_queue, d_feature, 0, 0, 0);

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
//	SAMPLE_CHECK_ERRORS(err);
	
	size_t global_work[3] = { n_points, 1 , 1 };
	/// Ke Wang adjustable local group size 2013/08/07 10:37:33
	size_t local_work_size= BLOCK_SIZE; // work group size is defined by RD_WG_SIZE_0 or RD_WG_SIZE_0_0 2014/06/10 17:00:51
	if(global_work[0]%local_work_size !=0)
	  global_work[0]=(global_work[0]/local_work_size+1)*local_work_size;

	err = clEnqueueNDRangeKernel(cmd_queue, kernel2, 1, NULL, global_work, &local_work_size, 0, 0, 0);
	// err = clEnqueueNDRangeKernel(cmd_queue, kernel2, 1, NULL, global_work, &local_work_size, 0, 0, 0);

	if(err != CL_SUCCESS) { printf("ERROR1: clEnqueueNDRangeKernel()=>%d, globalsize: %d,localSize: %d,maxsize:%d failed\n", 
		err, global_work[0] * global_work[1], local_work_size, CL_DEVICE_MAX_WORK_ITEM_SIZES ); return -1; }
	
	membership_OCL = (int*) malloc(n_points * sizeof(int));
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

	free(membership_OCL);
}


int main( int argc, char** argv) 
{
	printf("WG size of kernel_swap = %d, WG size of kernel_kmeans = %d, work size = %d \n", BLOCK_SIZE, BLOCK_SIZE2,CL_DEVICE_MAX_WORK_GROUP_SIZE);

	setup(argc, argv);
	shutdown();
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
	
	size_t global_work[3] = { n_points, 1, 1 }; 

	/// Ke Wang adjustable local group size 2013/08/07 10:37:33
	size_t local_work_size=BLOCK_SIZE2; // work group size is defined by RD_WG_SIZE_1 or RD_WG_SIZE_1_0 2014/06/10 17:00:41
	if(global_work[0]%local_work_size !=0)
	  global_work[0]=(global_work[0]/local_work_size+1)*local_work_size;

	err = clEnqueueSVMMap(cmd_queue, CL_TRUE, CL_MAP_WRITE, d_cluster, n_clusters * n_features * sizeof(float), 0, 0, 0);
	//err &= clEnqueueWriteBuffer(cmd_queue, (cl_mem)d_cluster, 1, 0, n_clusters * n_features * sizeof(float), clusters[0], 0, 0, 0);
	memmove(d_cluster, clusters[0], n_clusters * n_features * sizeof(float));
	err &= clEnqueueSVMUnmap(cmd_queue, d_cluster, 0, 0, 0);
	
	if(err != CL_SUCCESS) { printf("ERROR: clEnqueueWriteBuffer d_cluster (size:%d) => %d\n", n_points, err); return -1; }

	int size = 0; int offset = 0;
					
	clSetKernelArgSVMPointer(kernel_s, 0, (void*) d_feature_swap);
	clSetKernelArgSVMPointer(kernel_s, 1, (void*) d_cluster);
	clSetKernelArgSVMPointer(kernel_s, 2, (void*) d_membership);
	clSetKernelArg(kernel_s, 3, sizeof(cl_int), (void*) &n_points);
	clSetKernelArg(kernel_s, 4, sizeof(cl_int), (void*) &n_clusters);
	clSetKernelArg(kernel_s, 5, sizeof(cl_int), (void*) &n_features);
	clSetKernelArg(kernel_s, 6, sizeof(cl_int), (void*) &offset);
	clSetKernelArg(kernel_s, 7, sizeof(cl_int), (void*) &size);

	err = clEnqueueNDRangeKernel(cmd_queue, kernel_s, 1, NULL, global_work, &local_work_size, 0, 0, 0);
	// err = clEnqueueNDRangeKernel(cmd_queue, kernel_s, 1, NULL, global_work, &local_work_size, 0, 0, 0);

	if(err != CL_SUCCESS) { printf("ERROR2: clEnqueueNDRangeKernel()=>%d, 2, %d failed\n", err, local_work_size); return -1; }
	clFinish(cmd_queue);

	err = clEnqueueSVMMap(cmd_queue, CL_TRUE, CL_MAP_WRITE, d_membership, n_points * sizeof(int), 0, 0, 0);
//	err = clEnqueueReadBuffer(cmd_queue, d_membership, 1, 0, n_points * sizeof(int), membership_OCL, 0, 0, 0);
	memmove(membership_OCL,d_membership, n_points * sizeof(int));
	err &= clEnqueueSVMUnmap(cmd_queue, d_membership, 0, 0, 0);

	if(err != CL_SUCCESS) { printf("ERROR: Memcopy Out\n"); return -1; }
	
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

	return delta;
}
