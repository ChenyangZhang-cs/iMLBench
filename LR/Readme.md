# Presentation

Iterative and parallelized execution time comparison, for the calculus of a regression linear model. Use of two datasets :

* With __545 data__, price per surface of an appartment location
* With __96453 data__, atmospheric pressure per temperature

Using of OpenCL to have real, hardware, parallelization.

## How to use

1. Fetch the repository
2. Into the directory, tape __make__
3. Tape __make term__, to have result on terminal
4. Tape __make graph__, to have result on graphics

## OpenCL information

* __Platform__ vendor specific opencl implementation
* __Context__ collection of devices able to use OpenCL, they work together
* __Devices__ physical things that run OpenCL code (CPU/GPU/Accelerator)
* __Host__ client side, calling code
* __Kernel__ function that which does the work
* __Work item__ instance of kernel that done a bit of the work
* __Work group__ collection of work items
* __Command queue__ way for host to communicate to device, send it commands to execute
* __Memory__ local/global/private/constant
* __Buffer__ area of memory on OpenGL device
* __Compute unit__ work group and its associated local memory

![Memory model](https://www.researchgate.net/profile/Ruben_Salvador2/publication/319285029/figure/fig1/AS:573204617850880@1513674038094/OpenCL-model-From-https-wwwkhronosorg.png)

### Kernel

__kernel void functionName(__global int* inData) {}

* Have a *__kernel* prefix
* Return void
* Take at least one argument
* __get_global_id()__ uniquely identify each work item executing the kernel
* __get_local_id()__ uniquely identify each work item in a work group
