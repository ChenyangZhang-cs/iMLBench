## 3D Convolution Neural Network (3DCNN)

### Application Introduction

3D Convolution Neural Network (3DCNN) is the feed-forward step of a fully-connected CNN with 3-dimension input. The neurons of 3DCNN are arranged in three dimensions: width, height, and depth. Since 3DCNN has another dimension compared with 2DCNN, it can be applied in training spatial structured data, e.g., video and medical images. We implement 3DCNN based on [“Polybench: The polyhedral benchmark suite,” URL: http://www.cs.ucla.edu/pouchet/software/polybench, 2012].


### Usage

- using `bash run.sh` to run application with CPU offset from 0 to 100, spaced by 10.

- 3D [cpu_offset]

    cpu_offset is an integer which represents the CPU offset of CPU-GPU co-running. 10 means CPU executes 10% computational task. (default: 0)

    example:
    $ ./3D 10

### Expected Result

```
CPU offset: xx
Total time: xxx ms
Error Threshold of 1.05 Percent: 0
```

