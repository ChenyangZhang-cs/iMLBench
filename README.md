# iMLBench

iMLBench is a machine learning benchmark suite targeting  CPU-GPU integrated architectures.

## Programs

Currently, we provide machine learning workloads including Linear Regression (LR), K-means (KM), K Nearest Neighbor (KNN), Back Propagation (BP), 2D Convolution Neural Network (2DCNN), 3D Convolution Neural Network (3DCNN), Multi-layer Perceptron (MLP), and Winograd Convolution (Winograd).

## Implementation and Reproduction

We provide OpenMP and OpenCL implementations for CPU-only, GPU-only, CPU-GPU co-running common machine learning applications. 

The code we provide is for eight machine learning applications of  *iMLBench* and *iMLBench w/o zero-copy.* 
All experiments in *Section 5 Experiment* can be conducted with our code uploaded to Code Ocean. 
For Figure 5,
we use Microarchitecture-Independent Workload Characterization (MICA) framework developed by Hoste and Eeckhout (\url{https://github.com/boegel/MICA}) to measure the eight applications in the directory of `code/` in the *Core Files* category of our project in Code Ocean.
For Figure 6,
we measure the execution time of the eight applications and calculate the speedup.
For Figure 7,
after obtaining the execution time of baseline and zero-copy, denoted as $t_{baseline}$ and $t_{zerocopy}$,
we calculate the benefits by $(t_{baseline} - t_{zerocopy})/t_{baseline}$.
 Figure 8 is the time measurement of the applications.
 Figure 9 is the normalized performance divided by the price, and the price has been shown in Table 1 of the paper.
 Please read our bash file `run.sh`.
We use Excel to draw all figures in the paper and upload the Excel file `figures.xlsm`, including experimental results and figures, in the *Other Files* category of our project in Code Ocean.

The co-running programs corresponding to our paper depend on OpenMP and OpenCL 2.0 and must be executed in an integrated architecture to reproduce the results in our paper. Since Code Ocean does not support OpenCL 2.0, we additionally provide a CUDA CPU-GPU co-running example, KNN, which can be complied and executed in Code Ocean. We use this example to demonstrate that our code is correct and executable as well as show our co-running design. However, this CUDA example cannot produce the experimental results in our paper since the CUDA program running on the discrete architectures of Code Ocean cannot realize our special design including zero-copy and co-running for the integrated architectures, as shown in paper.

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