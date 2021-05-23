## 2D Convolution Neural Network (2DCNN)

### Application Introduction

2D Convolution Neural Network (2DCNN) is the feed-forward step of a fully-connected convolution neural network (CNN) with 2-dimension input. CNN is a class of deep neural networks and is used extensively in image and video recognition, image classification, recommender systems, medical image analysis, natural language processing, and financial time series. The original code is from Polybench suite developed by Louis-Noel Poucht. Note that we do not include back propagation of 2DCNN and 3DCNN because of the existing of BP. The original code is from [“Understanding co-running behaviors on integrated CPU/GPU architectures,” TPDS, 2017]

### Usage

- using `bash run.sh` to run application with CPU offset from 0 to 100, spaced by 10.

- 2D [cpu_offset]

    cpu_offset is an integer which represents the CPU offset of CPU-GPU co-running. 10 means CPU executes 10% computational task. (default: 0)

    example:
    $ ./2D 10

### Expected Result

```
CPU offset: xx
Total time: xxx ms
Error Threshold of 1.05 Percent: 0
```

