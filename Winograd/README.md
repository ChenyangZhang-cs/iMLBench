## Winograd Convolution (Winograd)

### Application Introduction

Winograd Convolution (Winograd) is an optimized convolution operation for 3×3 kernels. By employing Winograd minimal filtering algorithm, Winograd convolution reduces the multiplication complexity by at least 2.25× compared to the original implementation, which shows great success in accelerating CNNs, such as VGG and ResNet.

### Usage

- using `bash run.sh` to run application with CPU offset from 0 to 100, spaced by 10.

- WinogradConv2D [cpu_offset]

    cpu_offset is an integer which represents the CPU offset of CPU-GPU co-running. 10 means CPU executes 10% computational task. (default: 0)

    example:
    $ ./WinogradConv2D 10

### Expected Result

```
CPU offset: xx
Total time: xxx ms
Error Threshold of 1.05 Percent: 0
```

