# iMLBench

iMLBench is a machine learning benchmark suite targeting  CPU-GPU integrated architectures.

## Programs

Currently, we provide machine learning workloads including Linear Regression (LR), K-means (KM), K Nearest Neighbor (KNN), Back Propagation (BP), 2D Convolution Neural Network (2DCNN), 3D Convolution Neural Network (3DCNN), Multi-layer Perceptron (MLP), and Winograd Convolution (Winograd).

## Implementation and Reproduction

We provide OpenMP and OpenCL implementations for CPU-only, GPU-only, CPU-GPU co-running common machine learning applications. 

The code we provide is for eight machine learning applications of  *iMLBench* and Winograd of *iMLBench w/o zero-copy.* Since *iMLBench w/o zero-copy* is not the focus of our work, we just provide one example application Winograd. All experimental results of *iMLBench* and *iMLBench w/o co-runing* (setting the CPU offset to 0) can be reproduced by this code, including *iMLBench* and *iMLBench w/o co-runing* in Figure 6, the black lines in Figure 8, and results of the APU and the CPU in Figure 9. For details, the results of *iMLBench* can be produced by running our code with provided `run.sh`, because `run.sh` runs the applications with CPU offset from 0 to 100, spaced by 10. The results of *iMLBench w/o co-runing* is the results when the CPU offset is 0. 

The co-running programs corresponding to our paper depend on OpenMP and OpenCL 2.0. Since Code Ocean doesn't support OpenCL 2.0, we additionally provide a CUDA CPU-GPU co-running example application, KNN, which can be complied and executed in Code Ocean, to demenstrate our co-running design. However, this CUDA example cannot produce the experimental results in our paper since CUDA program cannot realize our special design for the integrated architectures proposed in paper.

## Analysis of Possible Different Reproduced Results

We analyze some factors which may affect the reproduced results.

- Programming language. The CUDA KNN we provide cannot reproduce the experimental results in our paper since CUDA program cannot realize our special design for the integrated architectures proposed in paper. OpenMP and OpenCL 2.0 are strongly recommended to reproduce our results.
- Computational capacity of the experimental platform. If the capacity of the GPU and the CPU in the integrated architecture is close, the co-running design may get big benefits. On the contrary, if the computational capacity of the GPU is much higher than the CPU, the co-running design may get little benefits due to the huge capacity gap. 
- Hardware design of the shared memory of the experimental platform. The shared memory design between the CPU and the GPU of the integrated architecture affects the performance of the zero-copy design in iMLBench.

## Paper

Our related paper, ''iMLBench: A Machine Learning Benchmark Suite for CPU-GPU Integrated Architectures'', can be downloaded from TPDS (https://ieeexplore.ieee.org/document/9305972). If you use our benchmark, please cite our paper:

```
@ARTICLE{9305972,  
    author={C. {Zhang} and F. {Zhang} and X. {Guo} and B. {He} and X. {Zhang} and X. {Du}},  
    journal={IEEE Transactions on Parallel and Distributed Systems},   
    title={{iMLBench: A Machine Learning Benchmark Suite for CPU-GPU Integrated Architectures}},   
    year={2020},  
    volume={},  
    number={},  
    pages={1-1},  
    doi={10.1109/TPDS.2020.3046870}
}
```

## Acknowledgement

If you have problems, please contact Chenyang Zhang:

* [chenyangzhang@ruc.edu.cn](chenyangzhang@ruc.edu.cn)


Thanks for your interests in iMLBench and hope you like it. (^_^)