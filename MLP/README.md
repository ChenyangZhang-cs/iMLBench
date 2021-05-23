## Multi-layer Perceptron (MLP)

### Application Introduction

Multi-layer Perceptron (MLP) is a typical class of feed-forward artificial neural network, comprising of an input layer, several hidden layers, and an output layer. Since the neurons within MLP use nonlinear activation functions, MLP can deal with nonlinear data. MLP is often used to get approximate solutions for complicated problems and to create regression models and classifications. Our implementation is based on ClNET [“clNET: OpenCL for Nets,” https://github.com/ mz24cn/clnet, 2018].


### Usage

- using `bash run.sh` to run application with CPU offset from 0 to 100, spaced by 10.

- build/OpenCLNet MLP /off [int]

    /off            cpu_offset, an integer which represents the CPU offset of CPU-GPU co-running. 10 means CPU executes 10% computational task. (default: 0)

    example:
    $ ./build/OpenCLNet MLP /off 10

### Expected Result

```
CPU offset: xx
Error rate: xxx
Total time: xxx ms
```