#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main() {
    int i, j;
    char info[1024];

    cl_int status;
    cl_uint nPlatform;
    cl_platform_id* listPlatform;
    cl_uint nDevice;
    cl_device_id* listDevice;
    clGetPlatformIDs(0, NULL, &nPlatform);
    printf("Platform number: %d\n", nPlatform);
    listPlatform = (cl_platform_id*)malloc(nPlatform * sizeof(cl_platform_id));
    clGetPlatformIDs(nPlatform, listPlatform, NULL);

    for (i = 0; i < nPlatform; i++) {
        clGetPlatformInfo(listPlatform[i], CL_PLATFORM_NAME, 1024, info, NULL);
        printf("Platform[%d]:\n\tName\t\t%s", i, info);
        clGetPlatformInfo(listPlatform[i], CL_PLATFORM_VERSION, 1024, info, NULL);
        printf("\n\tVersion\t\t%s", info);
        //clGetPlatformInfo(listPlatform[i], CL_PLATFORM_VENDOR, 1024, info, NULL);
        //printf("\n\tVendor\t\t%s", info);
        //clGetPlatformInfo(listPlatform[i], CL_PLATFORM_PROFILE, 1024, info, NULL);
        //printf("\n\tProfile\t\t%s", info);
        clGetPlatformInfo(listPlatform[i], CL_PLATFORM_EXTENSIONS, 1024, info, NULL);
        printf("\n\tExtension\t%s", info);

        clGetDeviceIDs(listPlatform[i], CL_DEVICE_TYPE_ALL, 0, NULL, &nDevice);
        listDevice = (cl_device_id*)malloc(nDevice * sizeof(cl_device_id));
        clGetDeviceIDs(listPlatform[i], CL_DEVICE_TYPE_ALL, nDevice, listDevice, NULL);

        for (j = 0; j < nDevice; j++) {
            printf("\n");
            clGetDeviceInfo(listDevice[j], CL_DEVICE_NAME, 1024, info, NULL);
            printf("\n\tDevice[%d]:\n\tName\t\t%s", j, info);
            clGetDeviceInfo(listDevice[j], CL_DEVICE_VERSION, 1024, info, NULL);
            printf("\n\tVersion\t\t%s", info);
            clGetDeviceInfo(listDevice[j], CL_DEVICE_TYPE, 1024, info, NULL);
            switch (info[0]) {
                case CL_DEVICE_TYPE_DEFAULT:
                    strcpy(info, "DEFAULT");
                    break;
                case CL_DEVICE_TYPE_CPU:
                    strcpy(info, "CPU");
                    break;
                case CL_DEVICE_TYPE_GPU:
                    strcpy(info, "GPU");
                    break;
                case CL_DEVICE_TYPE_ACCELERATOR:
                    strcpy(info, "ACCELERATOR");
                    break;
                case CL_DEVICE_TYPE_CUSTOM:
                    strcpy(info, "CUSTOM");
                    break;
                case CL_DEVICE_TYPE_ALL:
                    strcpy(info, "ALL");
                    break;
            }
            printf("\n\tType\t\t%s", info);

            cl_device_svm_capabilities svm;
            clGetDeviceInfo(listDevice[j], CL_DEVICE_VERSION, sizeof(cl_device_svm_capabilities), &svm, NULL);
            info[0] = '\0';
            if (svm & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER)
                strcat(info, "COARSE_GRAIN_BUFFER ");
            if (svm & CL_DEVICE_SVM_FINE_GRAIN_BUFFER)
                strcat(info, "FINE_GRAIN_BUFFER ");
            if (svm & CL_DEVICE_SVM_FINE_GRAIN_SYSTEM)
                strcat(info, "FINE_GRAIN_SYSTEM ");
            if (svm & CL_DEVICE_SVM_ATOMICS)
                strcat(info, "ATOMICS");
            printf("\n\tSVM\t\t%s", info);
        }
        printf("\n\n");
        free(listDevice);
    }
    free(listPlatform);
    getchar();
    return 0;
}