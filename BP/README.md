## Back Propagation (BP)

### Application Introduction

Back Propagation (BP) is a widely used algorithm to adjust weights during the training process of neural networks. It first generates random weights and calculates the output layer, and then alters the weights with the gradient of the loss to weights calculated by the chain rule. Our implementation, based on Rodinia’s implementation, propagates backward from the last layer to avoid redundant computing in the chain rule. The original BP is from [“Understanding co-running behaviors on integrated CPU/GPU architectures,” TPDS, 2017]

### Usage

- using `bash run.sh` to run the application with CPU offset from 0 to 100, spaced by 10.

- backprop [num_of_input_elements] [cpu_offset]
    
    num_of_input_elements must be divided by 16. (default: 0)

    cpu_offset is an integer which represents the CPU offset of CPU-GPU co-running. 10 means CPU executes 10% computational task. (default: 0)

    example:
    $ ./backprop 819200 10

### Expected Result

```
CPU offset: xx
Total time: xxx ms
```

